import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "k12_superintendents_caep_2026"
TASK_DESCRIPTION = """Identify 3 current K-12 public school district superintendents in the United States who meet all of the following criteria:

For each superintendent:
1. Holds an EdD (Doctor of Education) in Educational Leadership, Educational Administration, or a closely related education field
2. The EdD was earned from a university that currently holds CAEP (Council for the Accreditation of Educator Preparation) accreditation for its educator preparation programs
3. Holds a Master's degree in education or a related field from an accredited institution
4. Currently serving as superintendent of a K-12 public school district (not as interim superintendent)
5. Has been serving in the current superintendent position for 4 to 7 years as of March 2026
6. Previously held at least one administrative leadership position in K-12 education (such as principal, assistant principal, director, or assistant superintendent) before becoming superintendent
7. Currently leads a district with student enrollment of at least 35,000 students
8. The district is located in one of the following states: California, Texas, Florida, New York, or Illinois
9. The district's state requires administrative certification or licensure for superintendents
10. Has a publicly accessible professional biography on the school district's official website that includes information about their educational background and professional experience
"""

ALLOWED_STATES = ["California", "Texas", "Florida", "New York", "Illinois"]
WINDOW_MIN_YEARS = 4
WINDOW_MAX_YEARS = 7
REFERENCE_MONTH_YEAR = "March 2026"
VALID_START_YEAR_RANGE = list(range(1990, 2027))  # for parsing sanity


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DegreeInfo(BaseModel):
    degree_name: Optional[str] = None  # e.g., "EdD", "Ed.D.", "Doctor of Education", "M.Ed.", "MA"
    field: Optional[str] = None        # e.g., "Educational Leadership"
    institution: Optional[str] = None  # e.g., "University of X"
    completion_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # URLs supporting this degree


class SuperintendentItem(BaseModel):
    # Identity and district
    name: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None

    # Biography URL (official district website page about the superintendent)
    biography_url: Optional[str] = None

    # Education
    edd: Optional[DegreeInfo] = None
    masters: Optional[DegreeInfo] = None

    # Specific accreditation sources for CAEP status of the EdD institution
    caep_accreditation_sources: List[str] = Field(default_factory=list)

    # Current role and tenure
    current_position_sources: List[str] = Field(default_factory=list)  # URLs explicitly stating current superintendent role
    start_date: Optional[str] = None     # free text like "July 2019", "2019", "August 2021"
    start_year: Optional[str] = None     # if explicitly mentioned as year
    start_sources: List[str] = Field(default_factory=list)  # URLs supporting start date/year

    # Prior administrative leadership roles
    prior_admin_roles: List[str] = Field(default_factory=list)  # e.g., ["principal", "assistant superintendent"]
    prior_admin_sources: List[str] = Field(default_factory=list)

    # District size (enrollment)
    district_enrollment_claim: Optional[str] = None  # free text, e.g., "over 50,000 students"
    enrollment_sources: List[str] = Field(default_factory=list)

    # District location
    location_sources: List[str] = Field(default_factory=list)

    # State superintendent certification/licensure requirement
    state_certification_sources: List[str] = Field(default_factory=list)


class SuperintendentsExtraction(BaseModel):
    superintendents: List[SuperintendentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendents() -> str:
    return f"""
Extract up to three (3) K-12 public school district superintendents exactly as presented in the answer. If the answer contains more than 3, return only the first 3. If fewer than 3, return as many as available, and fill missing fields with null or empty arrays as appropriate.

For each superintendent, extract the following fields as a JSON object:

- name: Full name of the superintendent.
- district: The district they lead.
- state: The U.S. state of the district (full state name).
- biography_url: The URL to the superintendent's official district biography page (must be on the district's official website domain). If multiple bios are given, prefer the district's official site; otherwise, pick one.

Education (degrees):
- edd: An object with:
  - degree_name: As written (e.g., "EdD", "Ed.D.", "Doctor of Education").
  - field: The field of study as written (e.g., "Educational Leadership", "Educational Administration", "Education Policy", etc.).
  - institution: Institution that awarded the EdD.
  - completion_year: As written if mentioned (e.g., "2019"); otherwise null.
  - sources: All URLs that directly support this EdD fact (include the biography URL if it supports this).
- caep_accreditation_sources: URLs that directly support the statement that the EdD-awarding institution CURRENTLY holds CAEP accreditation for its educator preparation programs (e.g., CAEP Accredited Provider Directory page, institution accreditation statement). Include all that are cited in the answer.

- masters: An object with:
  - degree_name: As written (e.g., "M.Ed.", "MA", "MS", "Master of Education", etc.).
  - field: Field of study if given (e.g., "Curriculum and Instruction").
  - institution: Institution name.
  - completion_year: As written if present; otherwise null.
  - sources: All URLs that directly support this master's degree (include the biography URL if it supports this).

Current role and tenure:
- current_position_sources: URLs that state they are currently superintendent (not interim) of the district (often the district leadership page or bio).
- start_date: Start date text as written (e.g., "July 2019", "2019", "August 2021"). If the answer mentions a specific start date or year, extract it here; otherwise null.
- start_year: A clean 4-digit year if explicitly present in the answer (e.g., "2019"), else null.
- start_sources: URLs supporting start date/year (include the bio if it states appointment year/date).

Prior administrative leadership:
- prior_admin_roles: A list of role titles they held BEFORE becoming superintendent (e.g., "principal", "assistant principal", "director", "assistant superintendent", "chief academic officer"). Use EXACT titles given in the answer text.
- prior_admin_sources: URLs supporting these roles (typically the bio).

District size and location:
- district_enrollment_claim: The student enrollment text as written (e.g., "serves about 45,000 students", "enrollment: 52,300"). If multiple are present, pick the most authoritative from the district or NCES website.
- enrollment_sources: URLs that support the enrollment figure/threshold.
- location_sources: URLs that support the district location and state.

State superintendent certification/licensure requirement:
- state_certification_sources: URLs that support that the STATE requires certification/licensure for superintendents (prefer official state DOE/BOE pages, statutes, or regulations).

IMPORTANT:
- Extract ONLY information explicitly present in the answer. Do NOT invent or infer.
- For all URL arrays, include every relevant URL mentioned in the answer for that criterion. If a URL is missing protocol, prepend "http://".
- If a required field is missing in the answer, return null (or an empty list for arrays).

Return a JSON object with a single key "superintendents" which is an array of up to 3 such superintendent objects as specified.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*args: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for arr in args:
        if not arr:
            continue
        for u in arr:
            if not u:
                continue
            url = u.strip()
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _safe_name(x: Optional[str]) -> str:
    return x if (x and x.strip()) else "the person"


def _safe_district(x: Optional[str]) -> str:
    return x if (x and x.strip()) else "the district"


def _safe_state(x: Optional[str]) -> str:
    return x if (x and x.strip()) else "the state"


def _extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", text)
    if not m:
        return None
    year = int(m.group(0))
    if year in VALID_START_YEAR_RANGE:
        return year
    return None


# --------------------------------------------------------------------------- #
# Verification for one superintendent                                         #
# --------------------------------------------------------------------------- #
async def verify_superintendent(
    evaluator: Evaluator,
    parent_node,
    sup: SuperintendentItem,
    idx: int,
) -> None:
    # Create superintendent parallel node
    sup_node = evaluator.add_parallel(
        id=f"Superintendent_{idx+1}",
        desc=f"The {'first' if idx==0 else 'second' if idx==1 else 'third'} superintendent meets all required criteria",
        parent=parent_node,
        critical=False,
    )

    name = _safe_name(sup.name)
    district = _safe_district(sup.district)
    state = _safe_state(sup.state)
    bio_url_list = [sup.biography_url] if sup.biography_url else []

    # 1) EdD Degree
    edd_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_EdD_Degree",
        desc="Holds an EdD in Educational Leadership, Educational Administration, or a closely related education field",
        parent=sup_node,
        critical=True,
    )
    edd_field = sup.edd.field if (sup.edd and sup.edd.field) else None
    edd_inst = sup.edd.institution if (sup.edd and sup.edd.institution) else None
    edd_sources = _merge_sources(bio_url_list, sup.edd.sources if sup.edd else None)
    edd_field_clause = f" in {edd_field}" if edd_field else ""
    edd_inst_clause = f" from {edd_inst}" if edd_inst else ""
    edd_claim = f"{name} holds an EdD (Doctor of Education){edd_field_clause}{edd_inst_clause}."
    await evaluator.verify(
        claim=edd_claim,
        node=edd_leaf,
        sources=edd_sources,
        additional_instruction=(
            "Accept equivalent phrasing such as 'Ed.D.' or 'Doctor of Education'. "
            "The field should be educational leadership/administration or a clearly related education field "
            "(e.g., education policy, K-12 leadership)."
        ),
    )

    # 2) CAEP Accreditation for EdD-awarding institution
    caep_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_CAEP_Accreditation",
        desc="The EdD was earned from a university that holds CAEP accreditation for its educator preparation programs",
        parent=sup_node,
        critical=True,
    )
    inst_for_caep = edd_inst or "the university where the EdD was earned"
    caep_sources = _merge_sources(sup.caep_accreditation_sources, sup.edd.sources if sup.edd else None)
    caep_claim = (
        f"{inst_for_caep} currently holds CAEP (Council for the Accreditation of Educator Preparation) "
        "accreditation for its educator preparation programs."
    )
    await evaluator.verify(
        claim=caep_claim,
        node=caep_leaf,
        sources=caep_sources,
        additional_instruction=(
            "Verify that the institution is listed as CAEP-accredited (e.g., in CAEP's Accredited Provider Directory "
            "or an official accreditation statement). The status should be current as of now. "
            "Do not accept outdated NCATE/TEAC references unless they explicitly indicate current CAEP accreditation."
        ),
    )

    # 3) Master's Degree
    masters_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Masters_Degree",
        desc="Holds a Master's degree in education or a related field from an accredited institution",
        parent=sup_node,
        critical=True,
    )
    m_field = sup.masters.field if (sup.masters and sup.masters.field) else None
    m_inst = sup.masters.institution if (sup.masters and sup.masters.institution) else None
    masters_sources = _merge_sources(bio_url_list, sup.masters.sources if sup.masters else None)
    m_field_clause = f" in {m_field}" if m_field else ""
    m_inst_clause = f" from {m_inst}" if m_inst else ""
    masters_claim = f"{name} holds a master's degree{m_field_clause}{m_inst_clause}."
    await evaluator.verify(
        claim=masters_claim,
        node=masters_leaf,
        sources=masters_sources,
        additional_instruction=(
            "Accept degree labels such as M.Ed., MSEd, MA (Education), MS (Education), or other closely related "
            "education-focused master's degrees. The institution should be accredited; if the page indicates regional "
            "or national accreditation, that is acceptable."
        ),
    )

    # 4) Current position (not interim)
    current_pos_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Current_Position",
        desc="Currently serving as superintendent of a K-12 public school district (not interim superintendent)",
        parent=sup_node,
        critical=True,
    )
    current_pos_sources = _merge_sources(bio_url_list, sup.current_position_sources)
    current_pos_claim = f"As of {REFERENCE_MONTH_YEAR}, {name} is the superintendent (not interim) of {district}."
    await evaluator.verify(
        claim=current_pos_claim,
        node=current_pos_leaf,
        sources=current_pos_sources,
        additional_instruction=(
            "Confirm the person currently holds the superintendent title at the district and is not an interim. "
            "Prefer the district's official pages (leadership, superintendent's office, or biography)."
        ),
    )

    # 5) Tenure duration 4–7 years as of March 2026
    tenure_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Tenure_Duration",
        desc="Has been serving in the current superintendent position for 4 to 7 years as of March 2026",
        parent=sup_node,
        critical=True,
    )
    start_year_from_field = _extract_year(sup.start_year) if sup.start_year else None
    start_year_from_text = _extract_year(sup.start_date) if sup.start_date else None
    start_year = start_year_from_field or start_year_from_text
    tenure_sources = _merge_sources(bio_url_list, sup.start_sources, sup.current_position_sources)
    if start_year:
        tenure_claim = (
            f"{name} began serving as superintendent of {district} in {start_year}, "
            f"which corresponds to between {WINDOW_MIN_YEARS} and {WINDOW_MAX_YEARS} years of service as of {REFERENCE_MONTH_YEAR}."
        )
    else:
        tenure_claim = (
            f"As of {REFERENCE_MONTH_YEAR}, {name} has served as superintendent of {district} "
            f"for a period between {WINDOW_MIN_YEARS} and {WINDOW_MAX_YEARS} years."
        )
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        sources=tenure_sources,
        additional_instruction=(
            "If a start year/date is present on the page, compute the tenure length as of March 2026. "
            "Pass if it is within 4–7 years inclusive (i.e., roughly 2019–2022 start years). "
            "If the page directly states a tenure length that implies 4–7 years as of 2026, that is acceptable."
        ),
    )

    # 6) Prior administrative leadership
    prior_admin_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Prior_Admin",
        desc="Previously held at least one administrative leadership position in K-12 education before becoming superintendent",
        parent=sup_node,
        critical=True,
    )
    prior_roles_list_text = ", ".join(sup.prior_admin_roles) if sup.prior_admin_roles else "at least one administrative leadership role in K-12"
    prior_admin_sources = _merge_sources(bio_url_list, sup.prior_admin_sources)
    prior_admin_claim = (
        f"Before becoming superintendent of {district}, {name} held {prior_roles_list_text} in K-12 education."
    )
    await evaluator.verify(
        claim=prior_admin_claim,
        node=prior_admin_leaf,
        sources=prior_admin_sources,
        additional_instruction=(
            "Qualifying roles include positions such as principal, assistant principal, director, department head, "
            "assistant/associate superintendent, chief academic officer, etc. Verify at least one such role is shown."
        ),
    )

    # 7) District size (>= 35,000 students)
    size_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_District_Size",
        desc="Currently leads a district with student enrollment of at least 35,000 students",
        parent=sup_node,
        critical=True,
    )
    size_sources = _merge_sources(sup.enrollment_sources)
    size_claim = f"The student enrollment of {district} is at least 35,000 students."
    await evaluator.verify(
        claim=size_claim,
        node=size_leaf,
        sources=size_sources,
        additional_instruction=(
            "Use enrollment figures found on the district website, state/national education data portals (e.g., NCES), "
            "or other authoritative sources. Accept phrasing like 'over 35,000', 'approximately 40,000', etc., "
            "as long as it clearly meets or exceeds 35,000."
        ),
    )

    # 8) State location (must be one of CA, TX, FL, NY, IL)
    state_loc_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_State_Location",
        desc="The district is located in California, Texas, Florida, New York, or Illinois",
        parent=sup_node,
        critical=True,
    )
    state_loc_sources = _merge_sources(bio_url_list, sup.location_sources, sup.current_position_sources)
    state_loc_claim = (
        f"{district} is located in {state}, and {state} is one of California, Texas, Florida, New York, or Illinois."
    )
    await evaluator.verify(
        claim=state_loc_claim,
        node=state_loc_leaf,
        sources=state_loc_sources,
        additional_instruction=(
            f"Confirm the district is in one of these states: {', '.join(ALLOWED_STATES)}. "
            "Use official district pages or other authoritative references provided."
        ),
    )

    # 9) State requires superintendent certification/licensure
    state_cert_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_State_Certification",
        desc="The district's state requires administrative certification or licensure for superintendents",
        parent=sup_node,
        critical=True,
    )
    state_cert_sources = _merge_sources(sup.state_certification_sources)
    state_cert_claim = f"The state of {state} requires administrative certification or licensure for superintendents."
    await evaluator.verify(
        claim=state_cert_claim,
        node=state_cert_leaf,
        sources=state_cert_sources,
        additional_instruction=(
            "Prefer state department of education, state board of education, or state statute/regulation pages. "
            "Verify that the state requires a certificate, license, or comparable credential specifically for the "
            "superintendent (or district-level superintendent) role."
        ),
    )

    # 10) Public biography on district site including education and experience
    bio_leaf = evaluator.add_leaf(
        id=f"S{idx+1}_Public_Biography",
        desc="Has a publicly accessible professional biography on the district's official website that includes information about educational background and professional experience",
        parent=sup_node,
        critical=True,
    )
    bio_claim = (
        f"There is a publicly accessible professional biography page on the official website of {district} for {name} "
        "that includes both their educational background and professional experience."
    )
    await evaluator.verify(
        claim=bio_claim,
        node=bio_leaf,
        sources=bio_url_list if bio_url_list else None,
        additional_instruction=(
            "Check that the page is on the district's official domain and that it clearly includes sections or text "
            "about the superintendent's education (degrees, institutions) and professional experience (roles, positions)."
        ),
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
        strategy=AggregationStrategy.PARALLEL,  # Root parallel, each superintendent evaluated independently
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

    # Build a top-level node mirroring rubric root (optional but keeps structure explicit)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify 3 K-12 public school district superintendents in the United States who meet all specified criteria",
        parent=root,
        critical=False,
    )

    # Extract structured superintendent info
    extracted = await evaluator.extract(
        prompt=prompt_extract_superintendents(),
        template_class=SuperintendentsExtraction,
        extraction_name="superintendents_extraction",
    )

    # Normalize to exactly 3 items (pad with empty entries if necessary)
    items: List[SuperintendentItem] = list(extracted.superintendents[:3])
    while len(items) < 3:
        items.append(SuperintendentItem())

    # Verify each superintendent against all criteria (parallel children under each superintendent node)
    for i in range(3):
        await verify_superintendent(evaluator, task_node, items[i], i)

    # Return summary with the constructed verification tree and extraction info
    return evaluator.get_summary()