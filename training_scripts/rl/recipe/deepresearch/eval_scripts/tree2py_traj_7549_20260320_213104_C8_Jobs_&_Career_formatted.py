import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "education_admins_tx_us_2026"
TASK_DESCRIPTION = """
Identify four educational administrators in the United States who meet the following criteria as of March 2026:

Administrator 1: Provide the full name and current position of the superintendent of the third-largest school district in Texas by enrollment. Include the name of the school district.

Administrator 2: Provide the full name of a superintendent of a Central Texas school district who became permanent superintendent in 2024 after previously serving as interim superintendent for that same district. Include the name of the school district and the date they became permanent superintendent.

Administrator 3: Provide the full name of an acting or current superintendent of a Texas school district who earned a bachelor's degree from a university located in Oklahoma. Include the name of the Texas school district, the name of the Oklahoma university, and the degree earned.

Administrator 4: Provide the full name of a university president who assumed office in 2025 after previously serving as president of another university in a different state. Include the name of both universities (current and previous), the state each university is located in, and the date they assumed the current presidency.

For each administrator, provide reference URL(s) that support your answer.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Admin1Data(BaseModel):
    district_name: Optional[str] = None
    superintendent_full_name: Optional[str] = None
    current_position: Optional[str] = None
    assumed_role_date: Optional[str] = None  # e.g., "January 1, 2024"
    enrollment_2024_2025_text: Optional[str] = None  # e.g., "~118,000", "about 118k", etc.
    sources: List[str] = Field(default_factory=list)


class Admin2Data(BaseModel):
    district_name: Optional[str] = None
    superintendent_full_name: Optional[str] = None
    interim_start_date: Optional[str] = None  # e.g., "January 9, 2023"
    permanent_date: Optional[str] = None  # e.g., "January 25, 2024"
    same_district_statement: Optional[str] = None  # text that indicates same district
    sources: List[str] = Field(default_factory=list)


class Admin3Data(BaseModel):
    district_name: Optional[str] = None
    superintendent_full_name: Optional[str] = None
    role_title: Optional[str] = None  # e.g., "acting superintendent"
    acting_named_date: Optional[str] = None  # e.g., "December 12, 2025"
    degree_name: Optional[str] = None  # e.g., "Bachelor of Science in Elementary Education"
    university_name: Optional[str] = None  # e.g., "Northeastern State University"
    university_state: Optional[str] = None  # e.g., "Oklahoma"
    sources: List[str] = Field(default_factory=list)


class Admin4Data(BaseModel):
    president_full_name: Optional[str] = None
    current_university_name: Optional[str] = None  # e.g., "Montana State University"
    current_university_state: Optional[str] = None  # e.g., "Montana"
    assumed_office_date: Optional[str] = None  # e.g., "July 1, 2025"
    current_university_ordinal: Optional[str] = None  # e.g., "13th"
    previous_university_name: Optional[str] = None  # e.g., "Northern Michigan University"
    previous_university_state: Optional[str] = None  # e.g., "Michigan"
    previous_role_title: Optional[str] = None  # e.g., "President"
    previous_assumed_date: Optional[str] = None  # e.g., "February 2023"
    previous_university_ordinal: Optional[str] = None  # e.g., "17th"
    sources: List[str] = Field(default_factory=list)


class AdminsExtraction(BaseModel):
    admin1: Optional[Admin1Data] = None
    admin2: Optional[Admin2Data] = None
    admin3: Optional[Admin3Data] = None
    admin4: Optional[Admin4Data] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_admins() -> str:
    return """
Extract the four administrators described in the task from the provided answer. Only extract what is explicitly present in the answer text. For each administrator, also extract the list of reference URLs the answer cites for that administrator.

Return a JSON object with fields admin1, admin2, admin3, admin4. For each:

admin1:
- district_name: the district explicitly named (e.g., "Cypress-Fairbanks ISD")
- superintendent_full_name: the full name (e.g., "Dr. Douglas Killian")
- current_position: the stated current position/title (e.g., "superintendent")
- assumed_role_date: the date they assumed the superintendent role (e.g., "January 1, 2024"); if not stated, null
- enrollment_2024_2025_text: any enrollment statement for 2024–2025 (e.g., "about 118,000 students"); null if not present
- sources: an array of all URLs cited in the answer for admin1; extract only valid URLs actually shown in the answer

admin2:
- district_name: the district explicitly named (e.g., "Austin ISD")
- superintendent_full_name: the full name (e.g., "Matias Segura")
- interim_start_date: the date they began interim superintendent (e.g., "January 9, 2023")
- permanent_date: the date they became permanent superintendent (e.g., "January 25, 2024")
- same_district_statement: the phrase/sentence indicating interim and permanent roles were in the same district (if present); else null
- sources: URLs cited for admin2

admin3:
- district_name: the Texas district explicitly named (e.g., "Leander ISD")
- superintendent_full_name: the full name (e.g., "Chris Clark")
- role_title: the role title stated (e.g., "acting superintendent")
- acting_named_date: the date they were named acting (e.g., "December 12, 2025")
- degree_name: the bachelor's degree text extracted (e.g., "Bachelor of Science in Elementary Education")
- university_name: the Oklahoma university (e.g., "Northeastern State University")
- university_state: the state text (should be "Oklahoma" if present)
- sources: URLs cited for admin3

admin4:
- president_full_name: the full name (e.g., "Brock Tessman")
- current_university_name: e.g., "Montana State University"
- current_university_state: e.g., "Montana"
- assumed_office_date: the date assumed the current presidency (e.g., "July 1, 2025")
- current_university_ordinal: ordinal or numbering if stated (e.g., "13th"); else null
- previous_university_name: e.g., "Northern Michigan University"
- previous_university_state: e.g., "Michigan"
- previous_role_title: title at the previous university (e.g., "president")
- previous_assumed_date: when they assumed that previous presidency (e.g., "February 2023")
- previous_university_ordinal: ordinal at the previous university if stated (e.g., "17th"); else null
- sources: URLs cited for admin4

General rules:
- If an item is not stated in the answer, set it to null (or [] for the URL list).
- Extract URLs exactly as shown (including protocol). Ignore malformed strings that are not URLs.
- Do not invent any information.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    uniq = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Basic validity check
        if not re.match(r"^https?://", u):
            continue
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return len(_normalize_urls(urls)) > 0


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_admin1(evaluator: Evaluator, parent_node, data: Optional[Admin1Data]) -> None:
    """
    Administrator 1:
    - Cypress-Fairbanks ISD (CFISD), third-largest TX district by enrollment
    - Enrollment approx 118,000 (2024–2025) is included in the answer
    - Superintendent: Dr. Douglas Killian; assumed office January 1, 2024; current superintendent as of Mar 2026
    - Include sources
    """
    node = evaluator.add_parallel(
        id="administrator_1",
        desc="Administrator 1 (per constraints): Superintendent of the third-largest Texas school district by enrollment (include full name, current position, district name, and supporting URLs).",
        parent=parent_node,
        critical=False
    )

    urls = _normalize_urls(data.sources if data else [])

    # Gating: minimal fields exist
    evaluator.add_custom_node(
        result=bool(data and data.district_name and data.superintendent_full_name),
        id="admin1_required_fields_present",
        desc="Administrator 1: Required fields (district and superintendent name) are present in the answer.",
        parent=node,
        critical=True
    )

    # Reference URLs presence
    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="admin1_reference_urls",
        desc="Provides reference URL(s) that support the Administrator 1 claims.",
        parent=node,
        critical=True
    )

    # Create leaves
    l_district = evaluator.add_leaf(
        id="admin1_district_is_cyfair",
        desc="Identifies the school district as Cypress-Fairbanks ISD.",
        parent=node,
        critical=True
    )
    l_third = evaluator.add_leaf(
        id="admin1_district_third_largest_tx",
        desc="States/establishes that Cypress-Fairbanks ISD is the third-largest school district in Texas by enrollment (relevant timeframe).",
        parent=node,
        critical=True
    )
    l_enroll = evaluator.add_leaf(
        id="admin1_enrollment_approx_118k_2024_2025",
        desc="Includes that Cypress-Fairbanks ISD enrollment is approximately 118,000 students as of 2024–2025.",
        parent=node,
        critical=True
    )
    l_name = evaluator.add_leaf(
        id="admin1_superintendent_is_douglas_killian",
        desc="Provides the superintendent's full name as Dr. Douglas Killian.",
        parent=node,
        critical=True
    )
    l_current = evaluator.add_leaf(
        id="admin1_current_position_superintendent",
        desc="Identifies Dr. Douglas Killian's current position as superintendent of Cypress-Fairbanks ISD (as of March 2026).",
        parent=node,
        critical=True
    )
    l_assumed = evaluator.add_leaf(
        id="admin1_assumed_role_jan_1_2024",
        desc="States that Dr. Douglas Killian assumed the superintendent role on January 1, 2024.",
        parent=node,
        critical=True
    )

    # Prepare claims
    claims = [
        (
            "The identified school district is Cypress-Fairbanks Independent School District (also known as Cypress-Fairbanks ISD or CFISD).",
            urls,
            l_district,
            "Treat 'Cypress-Fairbanks ISD', 'Cypress-Fairbanks Independent School District', and 'CFISD' as equivalent."
        ),
        (
            "Cypress-Fairbanks ISD is the third-largest school district in Texas by student enrollment around the 2024–2025 timeframe.",
            urls,
            l_third,
            "Allow near-timeframe sources (2023–2025) that clearly place CFISD as third in Texas by enrollment."
        ),
        (
            # This check is about inclusion in the answer text, not web evidence
            "The answer explicitly states that Cypress-Fairbanks ISD has approximately 118,000 students for the 2024–2025 school year (allow variants like 'about 118k', '~118,000').",
            None,
            l_enroll,
            "Judge only whether the answer text includes this information. Minor phrasing/format differences are acceptable."
        ),
        (
            "The superintendent is Dr. Douglas Killian.",
            urls,
            l_name,
            "Check that the cited source(s) identify Douglas (Doug) Killian as the superintendent of Cypress-Fairbanks ISD."
        ),
        (
            "As of March 2026, Dr. Douglas Killian is serving as superintendent of Cypress-Fairbanks ISD.",
            urls,
            l_current,
            "Accept if the page indicates he is the current/active superintendent with no end date. Do not require the page to mention 'March 2026' explicitly."
        ),
        (
            "Dr. Douglas Killian assumed the superintendent role at Cypress-Fairbanks ISD on January 1, 2024.",
            urls,
            l_assumed,
            "Allow minor date-format variants (e.g., 'Jan. 1, 2024')."
        ),
    ]

    await evaluator.batch_verify(claims)


async def verify_admin2(evaluator: Evaluator, parent_node, data: Optional[Admin2Data]) -> None:
    """
    Administrator 2:
    - Austin ISD in Central Texas
    - Superintendent: Matias Segura; interim start Jan 9, 2023; permanent Jan 25, 2024; same district
    - Include sources
    """
    node = evaluator.add_parallel(
        id="administrator_2",
        desc="Administrator 2 (per constraints): Central Texas superintendent who became permanent in 2024 after serving as interim superintendent for the same district (include full name, district, permanent-appointment date, and supporting URLs).",
        parent=parent_node,
        critical=False
    )

    urls = _normalize_urls(data.sources if data else [])

    evaluator.add_custom_node(
        result=bool(data and data.district_name and data.superintendent_full_name),
        id="admin2_required_fields_present",
        desc="Administrator 2: Required fields (district and superintendent name) are present in the answer.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="admin2_reference_urls",
        desc="Provides reference URL(s) that support the Administrator 2 claims.",
        parent=node,
        critical=True
    )

    l_dist = evaluator.add_leaf(
        id="admin2_district_is_austin_isd",
        desc="Identifies the school district as Austin ISD.",
        parent=node,
        critical=True
    )
    l_central = evaluator.add_leaf(
        id="admin2_district_in_central_texas",
        desc="States/establishes that Austin ISD is located in Central Texas.",
        parent=node,
        critical=True
    )
    l_name = evaluator.add_leaf(
        id="admin2_superintendent_is_matias_segura",
        desc="Provides the superintendent's full name as Matias Segura.",
        parent=node,
        critical=True
    )
    l_interim = evaluator.add_leaf(
        id="admin2_interim_start_jan_9_2023",
        desc="States that Matias Segura began serving as interim superintendent on January 9, 2023.",
        parent=node,
        critical=True
    )
    l_perm = evaluator.add_leaf(
        id="admin2_permanent_date_jan_25_2024",
        desc="Provides the specific date Matias Segura became permanent superintendent as January 25, 2024.",
        parent=node,
        critical=True
    )
    l_same = evaluator.add_leaf(
        id="admin2_interim_then_permanent_same_district",
        desc="Indicates the interim role and permanent appointment were for the same district.",
        parent=node,
        critical=True
    )

    claims = [
        (
            "The identified school district is Austin Independent School District (Austin ISD).",
            urls,
            l_dist,
            "Treat 'Austin ISD' and 'Austin Independent School District' as equivalent."
        ),
        (
            "Austin ISD is a school district located in Central Texas.",
            urls,
            l_central,
            "Allow sources that reasonably establish that Austin (and thus Austin ISD) is in the Central Texas region."
        ),
        (
            "The superintendent is Matias Segura.",
            urls,
            l_name,
            "Check that the source(s) identify Matias Segura as the (interim or permanent) superintendent of Austin ISD."
        ),
        (
            "Matias Segura began serving as interim superintendent on January 9, 2023.",
            urls,
            l_interim,
            "Allow minor date-format variants."
        ),
        (
            "Matias Segura became the permanent superintendent on January 25, 2024.",
            urls,
            l_perm,
            "Allow minor date-format variants."
        ),
        (
            "Both the interim and the permanent superintendent roles were at Austin ISD (the same district).",
            urls,
            l_same,
            "Confirm that both roles are for Austin ISD."
        ),
    ]

    await evaluator.batch_verify(claims)


async def verify_admin3(evaluator: Evaluator, parent_node, data: Optional[Admin3Data]) -> None:
    """
    Administrator 3:
    - Leander ISD (Texas)
    - Acting/current superintendent: Chris Clark; named acting Dec 12, 2025
    - Bachelor's degree (BS in Elementary Education) from Northeastern State University (Oklahoma)
    - Include sources
    """
    node = evaluator.add_parallel(
        id="administrator_3",
        desc="Administrator 3 (per constraints): Acting/current superintendent of a Texas school district who earned a bachelor's degree from a university in Oklahoma (include name, district, Oklahoma university, degree, and supporting URLs).",
        parent=parent_node,
        critical=False
    )

    urls = _normalize_urls(data.sources if data else [])

    evaluator.add_custom_node(
        result=bool(data and data.district_name and data.superintendent_full_name),
        id="admin3_required_fields_present",
        desc="Administrator 3: Required fields (district and superintendent name) are present in the answer.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="admin3_reference_urls",
        desc="Provides reference URL(s) that support the Administrator 3 claims.",
        parent=node,
        critical=True
    )

    l_dist = evaluator.add_leaf(
        id="admin3_district_is_leander_isd",
        desc="Identifies the school district as Leander ISD.",
        parent=node,
        critical=True
    )
    l_tx = evaluator.add_leaf(
        id="admin3_district_in_texas",
        desc="States/establishes that Leander ISD is located in Texas.",
        parent=node,
        critical=True
    )
    l_name = evaluator.add_leaf(
        id="admin3_superintendent_is_chris_clark",
        desc="Provides the superintendent's full name as Chris Clark.",
        parent=node,
        critical=True
    )
    l_role = evaluator.add_leaf(
        id="admin3_role_acting_superintendent",
        desc="Identifies Chris Clark as the acting superintendent of Leander ISD (as of March 2026).",
        parent=node,
        critical=True
    )
    l_named = evaluator.add_leaf(
        id="admin3_named_acting_dec_12_2025",
        desc="States that Chris Clark was named acting superintendent on December 12, 2025.",
        parent=node,
        critical=True
    )
    l_degree = evaluator.add_leaf(
        id="admin3_bachelors_degree_bs_elementary_ed",
        desc="Specifies that Chris Clark earned a Bachelor of Science degree in Elementary Education.",
        parent=node,
        critical=True
    )
    l_uni = evaluator.add_leaf(
        id="admin3_university_is_northeastern_state",
        desc="Provides the Oklahoma university name as Northeastern State University.",
        parent=node,
        critical=True
    )
    l_uni_ok = evaluator.add_leaf(
        id="admin3_university_located_in_oklahoma",
        desc="States/establishes that Northeastern State University is located in Oklahoma.",
        parent=node,
        critical=True
    )

    claims = [
        (
            "The identified school district is Leander Independent School District (Leander ISD).",
            urls,
            l_dist,
            "Treat 'Leander ISD' and 'Leander Independent School District' as equivalent."
        ),
        (
            "Leander ISD is located in Texas.",
            urls,
            l_tx,
            "Confirm that Leander ISD operates in the state of Texas."
        ),
        (
            "The (acting/current) superintendent is Chris Clark.",
            urls,
            l_name,
            "Accept 'acting superintendent' or 'superintendent' references tied to Chris Clark."
        ),
        (
            "Chris Clark is the acting superintendent of Leander ISD.",
            urls,
            l_role,
            "It's acceptable if the source indicates he is 'acting superintendent' without explicitly stating 'as of March 2026'."
        ),
        (
            "Chris Clark was named acting superintendent on December 12, 2025.",
            urls,
            l_named,
            "Allow minor date-format variants."
        ),
        (
            "Chris Clark earned a Bachelor of Science in Elementary Education.",
            urls,
            l_degree,
            "Allow phrasing such as 'B.S.' or 'Bachelor's in Elementary Education'."
        ),
        (
            "The bachelor's degree was from Northeastern State University.",
            urls,
            l_uni,
            "Confirm that Northeastern State University is the institution granting the bachelor's degree."
        ),
        (
            "Northeastern State University is located in Oklahoma.",
            urls,
            l_uni_ok,
            "Confirm that NSU is in Oklahoma."
        ),
    ]

    await evaluator.batch_verify(claims)


async def verify_admin4(evaluator: Evaluator, parent_node, data: Optional[Admin4Data]) -> None:
    """
    Administrator 4:
    - University president: Brock Tessman
    - Current: Montana State University (Montana), assumed July 1, 2025, 13th president
    - Previous: Northern Michigan University (Michigan), served as president, assumed Feb 2023, 17th president
    - Different states condition satisfied
    - Include sources
    """
    node = evaluator.add_parallel(
        id="administrator_4",
        desc="Administrator 4 (per constraints): University president who assumed office in 2025 after previously serving as president of another university in a different state (include names, states, dates, and supporting URLs).",
        parent=parent_node,
        critical=False
    )

    urls = _normalize_urls(data.sources if data else [])

    evaluator.add_custom_node(
        result=bool(data and data.president_full_name and data.current_university_name and data.previous_university_name),
        id="admin4_required_fields_present",
        desc="Administrator 4: Required fields (president name and current/previous universities) are present in the answer.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="admin4_reference_urls",
        desc="Provides reference URL(s) that support the Administrator 4 claims.",
        parent=node,
        critical=True
    )

    # Different states requirement (custom logic based on extracted states)
    evaluator.add_custom_node(
        result=bool(
            data
            and data.current_university_state
            and data.previous_university_state
            and isinstance(data.current_university_state, str)
            and isinstance(data.previous_university_state, str)
            and data.current_university_state.strip().lower() != data.previous_university_state.strip().lower()
        ),
        id="admin4_different_states_requirement",
        desc="Satisfies the 'different state' condition by indicating the current and previous universities are in different states.",
        parent=node,
        critical=True
    )

    # Leaves
    l_name = evaluator.add_leaf(
        id="admin4_president_is_brock_tessman",
        desc="Provides the university president's full name as Brock Tessman.",
        parent=node,
        critical=True
    )
    l_curr_uni = evaluator.add_leaf(
        id="admin4_current_university_montana_state",
        desc="Identifies the current university as Montana State University.",
        parent=node,
        critical=True
    )
    l_curr_state = evaluator.add_leaf(
        id="admin4_current_university_state_montana",
        desc="Provides/establishes that Montana State University is located in Montana.",
        parent=node,
        critical=True
    )
    l_assumed = evaluator.add_leaf(
        id="admin4_assumed_office_july_1_2025",
        desc="Provides the date Brock Tessman assumed the Montana State University presidency as July 1, 2025.",
        parent=node,
        critical=True
    )
    l_curr_ord = evaluator.add_leaf(
        id="admin4_is_13th_president_msu",
        desc="States that Brock Tessman became the 13th president of Montana State University.",
        parent=node,
        critical=True
    )
    l_prev_uni = evaluator.add_leaf(
        id="admin4_previous_university_northern_michigan",
        desc="Identifies the previous university as Northern Michigan University.",
        parent=node,
        critical=True
    )
    l_prev_state = evaluator.add_leaf(
        id="admin4_previous_university_state_michigan",
        desc="Provides/establishes that Northern Michigan University is located in Michigan.",
        parent=node,
        critical=True
    )
    l_prev_role = evaluator.add_leaf(
        id="admin4_previous_role_president_nmu",
        desc="States that Brock Tessman previously served as president of Northern Michigan University.",
        parent=node,
        critical=True
    )
    l_prev_assumed = evaluator.add_leaf(
        id="admin4_previous_assumed_feb_2023",
        desc="States that Brock Tessman assumed the Northern Michigan University presidency in February 2023.",
        parent=node,
        critical=True
    )
    l_prev_ord = evaluator.add_leaf(
        id="admin4_is_17th_president_nmu",
        desc="States that Brock Tessman served as the 17th president of Northern Michigan University.",
        parent=node,
        critical=True
    )

    claims = [
        (
            "The university president is Brock Tessman.",
            urls,
            l_name,
            "Confirm that the person identified is Brock Tessman."
        ),
        (
            "The current university is Montana State University.",
            urls,
            l_curr_uni,
            "Allow 'MSU' if it clearly refers to Montana State University."
        ),
        (
            "Montana State University is located in the U.S. state of Montana.",
            urls,
            l_curr_state,
            "Confirm MSU is in Montana."
        ),
        (
            "Brock Tessman assumed office as president of Montana State University on July 1, 2025.",
            urls,
            l_assumed,
            "Allow minor date-format variants."
        ),
        (
            "Brock Tessman is the 13th president of Montana State University.",
            urls,
            l_curr_ord,
            "Confirm the ordinal (13th) using the sources."
        ),
        (
            "The previous university is Northern Michigan University.",
            urls,
            l_prev_uni,
            "Confirm that his prior presidency was at Northern Michigan University."
        ),
        (
            "Northern Michigan University is located in the U.S. state of Michigan.",
            urls,
            l_prev_state,
            "Confirm NMU is in Michigan."
        ),
        (
            "Brock Tessman previously served as president of Northern Michigan University.",
            urls,
            l_prev_role,
            "Confirm that his role at NMU was president."
        ),
        (
            "Brock Tessman assumed the presidency of Northern Michigan University in February 2023.",
            urls,
            l_prev_assumed,
            "Allow the date to be any day in February 2023; exact day is not required."
        ),
        (
            "Brock Tessman served as the 17th president of Northern Michigan University.",
            urls,
            l_prev_ord,
            "Confirm the ordinal (17th) using the sources."
        ),
    ]

    await evaluator.batch_verify(claims)


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
    Evaluate an answer for the four educational administrators task.
    """
    # Initialize evaluator (root set to PARALLEL; non-critical to allow partial credit)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four educational administrators in the United States meeting the specified criteria as of March 2026, with supporting reference URL(s) for each (must also satisfy the provided constraints).",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_admins(),
        template_class=AdminsExtraction,
        extraction_name="admins_extraction"
    )

    # Optional: record expected entities for debugging/traceability
    evaluator.add_custom_info(
        info={
            "expected_entities": {
                "admin1": {
                    "district": "Cypress-Fairbanks ISD (CFISD)",
                    "superintendent": "Dr. Douglas Killian",
                    "assumed_role_date": "January 1, 2024",
                    "enrollment_2024_2025_approx": "≈118,000",
                    "rank_in_tx": "Third-largest"
                },
                "admin2": {
                    "district": "Austin ISD",
                    "superintendent": "Matias Segura",
                    "interim_start": "January 9, 2023",
                    "permanent_date": "January 25, 2024",
                    "region": "Central Texas"
                },
                "admin3": {
                    "district": "Leander ISD (Texas)",
                    "superintendent": "Chris Clark",
                    "role": "Acting superintendent",
                    "acting_named_date": "December 12, 2025",
                    "degree": "Bachelor of Science in Elementary Education",
                    "university": "Northeastern State University (Oklahoma)"
                },
                "admin4": {
                    "president": "Brock Tessman",
                    "current_university": "Montana State University (Montana)",
                    "assumed_office": "July 1, 2025",
                    "current_ordinal": "13th",
                    "previous_university": "Northern Michigan University (Michigan)",
                    "previous_assumed": "February 2023",
                    "previous_ordinal": "17th"
                }
            }
        },
        info_type="expectations",
        info_name="expected_entities"
    )

    # Build verification subtrees
    await verify_admin1(evaluator, root, extracted.admin1)
    await verify_admin2(evaluator, root, extracted.admin2)
    await verify_admin3(evaluator, root, extracted.admin3)
    await verify_admin4(evaluator, root, extracted.admin4)

    # Return summary
    return evaluator.get_summary()