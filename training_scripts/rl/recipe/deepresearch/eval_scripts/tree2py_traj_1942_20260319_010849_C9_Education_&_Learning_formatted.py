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
TASK_ID = "us_universities_4_combo_eval"
TASK_DESCRIPTION = """Identify four U.S. universities that meet the following specific combinations of institutional characteristics. For each university, provide its complete official name, a link to its official homepage, and URL references that verify each specified characteristic.

University 1: A university that holds Carnegie R1 classification (Very High Research Activity, 2025 designation), is located in the Southeastern United States (Alabama, Arkansas, Florida, Georgia, Kentucky, Louisiana, Mississippi, North Carolina, South Carolina, Tennessee, Virginia, or West Virginia), is a public institution, has total undergraduate enrollment between 15,000 and 30,000 students, and participates in NCAA Division I athletics.

University 2: A university that holds Carnegie R1 classification, is located in the Northeastern United States (Connecticut, Maine, Massachusetts, New Hampshire, New Jersey, New York, Pennsylvania, Rhode Island, or Vermont), is a private institution, has total undergraduate enrollment below 10,000 students, and holds regional accreditation from one of the seven recognized U.S. regional accrediting agencies.

University 3: A university that holds Carnegie R2 classification (High Research Activity, 2025 designation), is located in the Western United States (Alaska, Arizona, California, Colorado, Hawaii, Idaho, Montana, Nevada, New Mexico, Oregon, Utah, Washington, or Wyoming), is a public institution, has total undergraduate enrollment exceeding 20,000 students, and offers undergraduate engineering degree programs.

University 4: A university that holds Carnegie R1 classification, is located in the Midwestern United States (Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, or Wisconsin), is a public institution, offers undergraduate business degree programs, and has an acceptance rate below 25% for its most recent admitted class.

For each university, provide:
- The complete official name
- URL to the official university homepage
- URL reference confirming Carnegie classification (from Carnegie Classification website or institutional research page)
- URL reference confirming geographic location
- URL reference confirming public/private status
- URL reference for enrollment data (from Common Data Set, IPEDS, or official university fact sheet)
- URL reference for the fifth characteristic specific to that university (NCAA Division I status, regional accreditation, engineering program page, or business program page and acceptance rate data)
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityInfo(BaseModel):
    # Required identification
    name: Optional[str] = None
    homepage_url: Optional[str] = None

    # Common verification items
    carnegie_level: Optional[str] = None  # e.g., "R1", "R2", or full phrasing
    carnegie_source_urls: List[str] = Field(default_factory=list)

    state: Optional[str] = None  # e.g., "Florida" or "FL"
    location_source_urls: List[str] = Field(default_factory=list)

    institution_type: Optional[str] = None  # "public" or "private"
    type_source_urls: List[str] = Field(default_factory=list)

    undergrad_enrollment: Optional[str] = None  # keep as string to allow ranges or formatted numbers
    enrollment_source_urls: List[str] = Field(default_factory=list)

    # University-specific extra items
    # U1: NCAA Division I
    ncaa_division: Optional[str] = None
    ncaa_source_urls: List[str] = Field(default_factory=list)

    # U2: Regional accreditation
    accreditation_agency: Optional[str] = None
    accreditation_source_urls: List[str] = Field(default_factory=list)

    # U3: Engineering programs
    engineering_programs_url: Optional[str] = None
    engineering_source_urls: List[str] = Field(default_factory=list)

    # U4: Business programs + acceptance rate
    business_programs_url: Optional[str] = None
    business_source_urls: List[str] = Field(default_factory=list)
    acceptance_rate: Optional[str] = None  # e.g., "21%", "0.21", or textual
    acceptance_rate_source_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    university_1: Optional[UniversityInfo] = None
    university_2: Optional[UniversityInfo] = None
    university_3: Optional[UniversityInfo] = None
    university_4: Optional[UniversityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract structured information for FOUR distinct U.S. universities described in the answer. For each university, return:

    Common fields for all universities:
    - name: The COMPLETE official university name as written in the answer (string).
    - homepage_url: The official homepage URL (string).
    - carnegie_level: The Carnegie classification level stated in the answer (e.g., "R1: Very High Research Activity" or "R2: High Research Activity") (string).
    - carnegie_source_urls: List of URL(s) cited in the answer that confirm the Carnegie classification (from the official Carnegie Classification site or the university's institutional research/factbook page).
    - state: The U.S. state where the main campus is located (string). Use the state name or standard two-letter abbreviation exactly as the answer presents.
    - location_source_urls: List of URL(s) cited in the answer that confirm the geographic location (e.g., university "About" page, campus address page, contact page, or state profile page).
    - institution_type: "public" or "private" as stated in the answer.
    - type_source_urls: List of URL(s) cited in the answer that confirm public/private status (e.g., official "About" page, factbook, state system page).
    - undergrad_enrollment: The total undergraduate enrollment figure or range as written in the answer (string). Keep the original formatting; do not coerce into a number.
    - enrollment_source_urls: List of URL(s) cited in the answer that give undergraduate enrollment (prefer Common Data Set/IPEDS/official factbook).

    Additional fields by university:
    - University 1 (Southeast; NCAA Division I):
        - ncaa_division: The NCAA division listed in the answer (expect "Division I").
        - ncaa_source_urls: URL(s) that confirm the NCAA division (NCAA.org member directory or official athletics site).
    - University 2 (Northeast; regional accreditation; private; <10,000 undergrads):
        - accreditation_agency: The REGIONAL accrediting agency named in the answer (e.g., MSCHE, NECHE, HLC, SACSCOC, WSCUC, NWCCU, ACCJC).
        - accreditation_source_urls: URL(s) that confirm institutional REGIONAL accreditation (accreditor directory or university accreditation page).
    - University 3 (West; R2; public; >20,000 undergrads; engineering programs):
        - engineering_programs_url: A URL to the college/school of engineering or undergraduate engineering program page (if given).
        - engineering_source_urls: Any additional URL(s) confirming undergraduate engineering programs.
    - University 4 (Midwest; R1; public; business programs; acceptance rate <25%):
        - business_programs_url: A URL to the business school/undergraduate business program page (if given).
        - business_source_urls: Any additional URL(s) confirming undergraduate business programs.
        - acceptance_rate: The acceptance/admit rate value as written in the answer (string).
        - acceptance_rate_source_urls: URL(s) that report the acceptance/admit rate (prefer Common Data Set/institutional research; reputable ranking sites ok if official not provided).

    Output must be a JSON object with keys:
    - university_1: UniversityInfo
    - university_2: UniversityInfo
    - university_3: UniversityInfo
    - university_4: UniversityInfo

    IMPORTANT:
    - Only extract URLs explicitly present in the answer. Do not invent any URLs.
    - If a field is missing for a university, set it to null (or empty list for URL lists).
    - Preserve textual values verbatim from the answer (e.g., enrollment format like "about 24,000").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_url_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _is_filled(text: Optional[str]) -> bool:
    return bool(text and isinstance(text, str) and text.strip())


async def _verify_with_required_sources(
    evaluator: Evaluator,
    node_id: str,
    node_desc: str,
    parent,
    claim: str,
    urls: Optional[List[str]],
    add_ins: str,
    critical: bool = True,
) -> bool:
    """
    Convenience: create a leaf and verify the claim by URL(s).
    If no URLs provided, mark the leaf as failed immediately.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=critical
    )
    url_list = _non_empty_url_list(urls)
    if not url_list:
        # Explicitly fail due to missing sources for a source-grounded check
        leaf.score = 0.0
        leaf.status = "failed"
        return False

    return await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=url_list,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Verification subroutines for each university                                #
# --------------------------------------------------------------------------- #
async def verify_university_1(evaluator: Evaluator, root, u: UniversityInfo) -> None:
    """
    University 1:
      - Carnegie R1 (2025)
      - Southeastern US (states listed)
      - Public institution
      - Undergraduate enrollment between 15,000 and 30,000
      - NCAA Division I
    """
    uni_node = evaluator.add_sequential(
        id="University_1",
        desc="First university: Carnegie R1, Southeastern U.S., public, enrollment 15,000-30,000, NCAA Division I",
        parent=root,
        critical=False
    )

    # Step 1: Identification (Critical; both children critical)
    id_node = evaluator.add_parallel(
        id="U1_step1_identification",
        desc="Identify the university meeting all specified criteria",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_filled(u.name),
        id="U1_name_provided",
        desc="Complete official name of the university is provided",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_filled(u.homepage_url),
        id="U1_homepage_url",
        desc="URL to the university's official homepage is provided",
        parent=id_node,
        critical=True
    )

    # Step 2: Verification of properties (non-critical container)
    step2 = evaluator.add_parallel(
        id="U1_step2_verification",
        desc="Verify all institutional characteristics of the identified university",
        parent=uni_node,
        critical=False
    )

    # Carnegie verification
    carne = evaluator.add_parallel(
        id="U1_carnegie_verify",
        desc="Verify Carnegie R1 classification",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U1_carnegie_status",
        node_desc="University holds Carnegie R1: Very High Research Activity designation (2025 classification)",
        parent=carne,
        claim=f"{u.name} is classified as Carnegie R1 (Very High Research Activity) in the 2025 Carnegie Classification.",
        urls=u.carnegie_source_urls,
        add_ins="Prefer evidence from carnegieclassifications.acenet.edu or the university's official institutional research/factbook page explicitly stating R1 (Very High Research Activity). Accept common shorthand 'R1'.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U1_carnegie_source",
        node_desc="URL reference from Carnegie Classification website or institutional research page confirming R1 status",
        parent=carne,
        claim=f"The provided page explicitly confirms that {u.name} is an R1 (Very High Research Activity) university.",
        urls=u.carnegie_source_urls,
        add_ins="At least one URL should be carnegieclassifications.acenet.edu or an official *.edu institutional research/factbook page explicitly stating 'R1' or 'Very High Research Activity' for the institution.",
        critical=True
    )

    # Location verification (Southeast)
    loc = evaluator.add_parallel(
        id="U1_location_verify",
        desc="Verify location in Southeastern United States",
        parent=step2,
        critical=False
    )
    # State set membership is a logical check (no source needed); verify simply
    state_check_node = evaluator.add_leaf(
        id="U1_state_check",
        desc="State is one of: Alabama, Arkansas, Florida, Georgia, Kentucky, Louisiana, Mississippi, North Carolina, South Carolina, Tennessee, Virginia, West Virginia",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim=f"The state '{u.state}' is among the Southeastern United States list: Alabama, Arkansas, Florida, Georgia, Kentucky, Louisiana, Mississippi, North Carolina, South Carolina, Tennessee, Virginia, West Virginia.",
        node=state_check_node,
        additional_instruction="Allow standard two-letter USPS abbreviations (e.g., FL=Florida, GA=Georgia)."
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U1_location_source",
        node_desc="URL reference confirming location from official source",
        parent=loc,
        claim=f"The provided page explicitly states that {u.name} is located in the state of {u.state}.",
        urls=u.location_source_urls,
        add_ins="Look for city/state on 'About', Contact, or campus location pages. The page should clearly show the state.",
        critical=True
    )

    # Institutional type (Public)
    typ = evaluator.add_parallel(
        id="U1_type_verify",
        desc="Verify public institution status",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U1_public_status",
        node_desc="University is a public (state-funded) institution",
        parent=typ,
        claim=f"{u.name} is a public institution.",
        urls=u.type_source_urls,
        add_ins="Accept phrases like 'public research university', 'public university', or membership in a state university system.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U1_type_source",
        node_desc="URL reference confirming institutional type",
        parent=typ,
        claim=f"The provided page explicitly indicates that {u.name} is public.",
        urls=u.type_source_urls,
        add_ins="The page should clearly state 'public' or convey state-funded status.",
        critical=True
    )

    # Enrollment between 15,000 and 30,000 (undergraduate)
    enr = evaluator.add_parallel(
        id="U1_enrollment_verify",
        desc="Verify undergraduate enrollment between 15,000 and 30,000",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U1_enrollment_range",
        node_desc="Enrollment is between 15,000 and 30,000 students",
        parent=enr,
        claim="The total undergraduate enrollment is between 15,000 and 30,000 students.",
        urls=u.enrollment_source_urls,
        add_ins="Use official sources: Common Data Set (Section B), IPEDS, or the university's factbook. Ensure the figure is 'undergraduate', not total headcount including graduate.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U1_enrollment_source",
        node_desc="URL reference for enrollment data from Common Data Set, IPEDS, or official fact sheet",
        parent=enr,
        claim="This page provides an official figure for total undergraduate enrollment for the university.",
        urls=u.enrollment_source_urls,
        add_ins="Prefer CDS, IPEDS, or institution factbook/dashboards; the page should explicitly show undergraduate enrollment.",
        critical=True
    )

    # NCAA Division I
    ncaa = evaluator.add_parallel(
        id="U1_ncaa_verify",
        desc="Verify NCAA Division I athletic participation",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U1_ncaa_status",
        node_desc="NCAA Division I status is confirmed",
        parent=ncaa,
        claim=f"{u.name} competes in NCAA Division I.",
        urls=u.ncaa_source_urls,
        add_ins="Prefer NCAA.org member directory; the university's official athletics site is also acceptable if it clearly indicates 'NCAA Division I'.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U1_ncaa_source",
        node_desc="URL reference from NCAA.org or university athletics website",
        parent=ncaa,
        claim="This page clearly indicates NCAA Division I affiliation for the institution.",
        urls=u.ncaa_source_urls,
        add_ins="The page should explicitly mention 'NCAA Division I'.",
        critical=True
    )


async def verify_university_2(evaluator: Evaluator, root, u: UniversityInfo) -> None:
    """
    University 2:
      - Carnegie R1 (2025)
      - Northeastern US (states listed)
      - Private institution
      - Undergraduate enrollment < 10,000
      - Regional accreditation (one of seven regional agencies)
    """
    uni_node = evaluator.add_sequential(
        id="University_2",
        desc="Second university: Carnegie R1, Northeastern U.S., private, enrollment <10,000, regional accreditation",
        parent=root,
        critical=False
    )

    # Step 1: Identification
    id_node = evaluator.add_parallel(
        id="U2_step1_identification",
        desc="Identify the university meeting all specified criteria",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_filled(u.name),
        id="U2_name_provided",
        desc="Complete official name of the university is provided",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_filled(u.homepage_url),
        id="U2_homepage_url",
        desc="URL to the university's official homepage is provided",
        parent=id_node,
        critical=True
    )

    # Step 2: Verification
    step2 = evaluator.add_parallel(
        id="U2_step2_verification",
        desc="Verify all institutional characteristics of the identified university",
        parent=uni_node,
        critical=False
    )

    # Carnegie verification (R1)
    carne = evaluator.add_parallel(
        id="U2_carnegie_verify",
        desc="Verify Carnegie R1 classification",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U2_carnegie_status",
        node_desc="University holds Carnegie R1: Very High Research Activity designation (2025 classification)",
        parent=carne,
        claim=f"{u.name} is classified as Carnegie R1 (Very High Research Activity) in the 2025 Carnegie Classification.",
        urls=u.carnegie_source_urls,
        add_ins="Prefer evidence from carnegieclassifications.acenet.edu or an official *.edu institutional research/factbook page. Accept 'R1' phrasing.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U2_carnegie_source",
        node_desc="URL reference confirming R1 classification",
        parent=carne,
        claim=f"The provided page explicitly confirms that {u.name} is R1.",
        urls=u.carnegie_source_urls,
        add_ins="At least one URL should be Carnegie Classification or the university IR page explicitly stating R1.",
        critical=True
    )

    # Location verification (Northeast)
    loc = evaluator.add_parallel(
        id="U2_location_verify",
        desc="Verify location in Northeastern United States",
        parent=step2,
        critical=False
    )
    state_check_node = evaluator.add_leaf(
        id="U2_state_check",
        desc="State is one of: Connecticut, Maine, Massachusetts, New Hampshire, New Jersey, New York, Pennsylvania, Rhode Island, Vermont",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim=f"The state '{u.state}' is among the Northeastern United States list: Connecticut, Maine, Massachusetts, New Hampshire, New Jersey, New York, Pennsylvania, Rhode Island, Vermont.",
        node=state_check_node,
        additional_instruction="Allow standard two-letter USPS abbreviations (CT, ME, MA, NH, NJ, NY, PA, RI, VT)."
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U2_location_source",
        node_desc="URL reference confirming location",
        parent=loc,
        claim=f"The provided page explicitly states that {u.name} is located in the state of {u.state}.",
        urls=u.location_source_urls,
        add_ins="Look for city/state on 'About', Contact, or campus location pages.",
        critical=True
    )

    # Institutional type (Private)
    typ = evaluator.add_parallel(
        id="U2_type_verify",
        desc="Verify private institution status",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U2_private_status",
        node_desc="University is a private (not state-funded) institution",
        parent=typ,
        claim=f"{u.name} is a private institution.",
        urls=u.type_source_urls,
        add_ins="Accept phrasing like 'private research university' or 'private university'.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U2_type_source",
        node_desc="URL reference confirming institutional type",
        parent=typ,
        claim=f"The provided page explicitly indicates that {u.name} is private.",
        urls=u.type_source_urls,
        add_ins="The page should clearly state 'private'.",
        critical=True
    )

    # Enrollment < 10,000 undergrads
    enr = evaluator.add_parallel(
        id="U2_enrollment_verify",
        desc="Verify undergraduate enrollment below 10,000",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U2_enrollment_range",
        node_desc="Enrollment is below 10,000 students",
        parent=enr,
        claim="The total undergraduate enrollment is below 10,000 students.",
        urls=u.enrollment_source_urls,
        add_ins="Use official sources (CDS/IPEDS/factbook). Ensure the figure is undergraduate, not total including graduate.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U2_enrollment_source",
        node_desc="URL reference for enrollment data",
        parent=enr,
        claim="This page provides an official figure for total undergraduate enrollment for the university.",
        urls=u.enrollment_source_urls,
        add_ins="Prefer CDS, IPEDS, or institution factbook/dashboards.",
        critical=True
    )

    # Regional accreditation
    accr = evaluator.add_parallel(
        id="U2_accreditation_verify",
        desc="Verify regional accreditation status",
        parent=step2,
        critical=False
    )
    reg_agencies = "MSCHE, NECHE, HLC, SACSCOC, WSCUC, NWCCU, ACCJC"
    await _verify_with_required_sources(
        evaluator,
        node_id="U2_accreditation_status",
        node_desc="Regional accreditation from one of seven U.S. agencies is confirmed",
        parent=accr,
        claim=f"{u.name} is regionally accredited by one of the recognized agencies ({reg_agencies}).",
        urls=u.accreditation_source_urls,
        add_ins="Accept only regional institutional accreditors (MSCHE, NECHE, HLC, SACSCOC, WSCUC, NWCCU, ACCJC). Do NOT count programmatic/specialized accreditations (e.g., AACSB, ABET) for this check.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U2_accreditation_source",
        node_desc="URL reference confirming accreditation status",
        parent=accr,
        claim="This page explicitly confirms the university's regional accreditation and names the accrediting agency.",
        urls=u.accreditation_source_urls,
        add_ins="Prefer accreditor directories (e.g., MSCHE/NECHE/HLC/NWCCU/SACSCOC/WSCUC/ACCJC) or official university accreditation pages.",
        critical=True
    )


async def verify_university_3(evaluator: Evaluator, root, u: UniversityInfo) -> None:
    """
    University 3:
      - Carnegie R2 (2025)
      - Western US (states listed)
      - Public institution
      - Undergraduate enrollment > 20,000
      - Offers undergraduate engineering degree programs
    """
    uni_node = evaluator.add_sequential(
        id="University_3",
        desc="Third university: Carnegie R2, Western U.S., public, engineering program, enrollment >20,000",
        parent=root,
        critical=False
    )

    # Step 1: Identification
    id_node = evaluator.add_parallel(
        id="U3_step1_identification",
        desc="Identify the university meeting all specified criteria",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_filled(u.name),
        id="U3_name_provided",
        desc="Complete official name of the university is provided",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_filled(u.homepage_url),
        id="U3_homepage_url",
        desc="URL to the university's official homepage is provided",
        parent=id_node,
        critical=True
    )

    # Step 2: Verification
    step2 = evaluator.add_parallel(
        id="U3_step2_verification",
        desc="Verify all institutional characteristics of the identified university",
        parent=uni_node,
        critical=False
    )

    # Carnegie verification (R2)
    carne = evaluator.add_parallel(
        id="U3_carnegie_verify",
        desc="Verify Carnegie R2 classification",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U3_carnegie_status",
        node_desc="University holds Carnegie R2: High Research Activity designation (2025 classification)",
        parent=carne,
        claim=f"{u.name} is classified as Carnegie R2 (High Research Activity) in the 2025 Carnegie Classification.",
        urls=u.carnegie_source_urls,
        add_ins="Prefer evidence from carnegieclassifications.acenet.edu or official *.edu institutional research/factbook explicitly stating 'R2'/'High Research Activity'.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U3_carnegie_source",
        node_desc="URL reference confirming R2 classification",
        parent=carne,
        claim=f"The provided page explicitly confirms that {u.name} is R2 (High Research Activity).",
        urls=u.carnegie_source_urls,
        add_ins="At least one URL should clearly state 'R2' or 'High Research Activity'.",
        critical=True
    )

    # Location verification (West)
    loc = evaluator.add_parallel(
        id="U3_location_verify",
        desc="Verify location in Western United States",
        parent=step2,
        critical=False
    )
    state_check_node = evaluator.add_leaf(
        id="U3_state_check",
        desc="State is one of: Alaska, Arizona, California, Colorado, Hawaii, Idaho, Montana, Nevada, New Mexico, Oregon, Utah, Washington, Wyoming",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim=f"The state '{u.state}' is among the Western United States list: Alaska, Arizona, California, Colorado, Hawaii, Idaho, Montana, Nevada, New Mexico, Oregon, Utah, Washington, Wyoming.",
        node=state_check_node,
        additional_instruction="Allow standard two-letter USPS abbreviations (AK, AZ, CA, CO, HI, ID, MT, NV, NM, OR, UT, WA, WY)."
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U3_location_source",
        node_desc="URL reference confirming location",
        parent=loc,
        claim=f"The provided page explicitly states that {u.name} is located in the state of {u.state}.",
        urls=u.location_source_urls,
        add_ins="Look for city/state on 'About', Contact, or campus location pages.",
        critical=True
    )

    # Institutional type (Public)
    typ = evaluator.add_parallel(
        id="U3_type_verify",
        desc="Verify public institution status",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U3_public_status",
        node_desc="University is a public (state-funded) institution",
        parent=typ,
        claim=f"{u.name} is a public institution.",
        urls=u.type_source_urls,
        add_ins="Accept phrasing like 'public research university' or 'public university'.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U3_type_source",
        node_desc="URL reference confirming institutional type",
        parent=typ,
        claim=f"The provided page explicitly indicates that {u.name} is public.",
        urls=u.type_source_urls,
        add_ins="The page should clearly state 'public' or show state system affiliation.",
        critical=True
    )

    # Enrollment > 20,000 undergrads
    enr = evaluator.add_parallel(
        id="U3_enrollment_verify",
        desc="Verify undergraduate enrollment exceeds 20,000",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U3_enrollment_range",
        node_desc="Enrollment exceeds 20,000 students",
        parent=enr,
        claim="The total undergraduate enrollment exceeds 20,000 students.",
        urls=u.enrollment_source_urls,
        add_ins="Use official sources (CDS/IPEDS/factbook). Ensure the figure is undergraduate, not total including graduate.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U3_enrollment_source",
        node_desc="URL reference for enrollment data",
        parent=enr,
        claim="This page provides an official figure for total undergraduate enrollment for the university.",
        urls=u.enrollment_source_urls,
        add_ins="Prefer CDS, IPEDS, or institution factbook/dashboards.",
        critical=True
    )

    # Engineering programs (undergraduate)
    eng = evaluator.add_parallel(
        id="U3_engineering_verify",
        desc="Verify engineering program offering",
        parent=step2,
        critical=False
    )
    eng_urls = _non_empty_url_list(([u.engineering_programs_url] if _is_filled(u.engineering_programs_url) else []) + u.engineering_source_urls)
    await _verify_with_required_sources(
        evaluator,
        node_id="U3_engineering_exists",
        node_desc="Engineering degree programs are offered",
        parent=eng,
        claim="The university offers undergraduate engineering degree programs (e.g., B.S. in Engineering or B.S. in Mechanical/Electrical/Civil/etc.).",
        urls=eng_urls,
        add_ins="Look for a College/School of Engineering page or undergraduate engineering majors list.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U3_engineering_source",
        node_desc="URL reference to engineering college/program page",
        parent=eng,
        claim="This page is an official engineering college/school or undergraduate engineering program page for the university.",
        urls=eng_urls,
        add_ins="Accept *.edu pages clearly tied to the institution and describing its engineering degrees/majors.",
        critical=True
    )


async def verify_university_4(evaluator: Evaluator, root, u: UniversityInfo) -> None:
    """
    University 4:
      - Carnegie R1 (2025)
      - Midwestern US (states listed)
      - Public institution
      - Offers undergraduate business programs
      - Acceptance rate below 25% (most recent admitted class)
    """
    uni_node = evaluator.add_sequential(
        id="University_4",
        desc="Fourth university: Carnegie R1, Midwestern U.S., public, business program, acceptance rate <25%",
        parent=root,
        critical=False
    )

    # Step 1: Identification
    id_node = evaluator.add_parallel(
        id="U4_step1_identification",
        desc="Identify the university meeting all specified criteria",
        parent=uni_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_filled(u.name),
        id="U4_name_provided",
        desc="Complete official name of the university is provided",
        parent=id_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_filled(u.homepage_url),
        id="U4_homepage_url",
        desc="URL to the university's official homepage is provided",
        parent=id_node,
        critical=True
    )

    # Step 2: Verification
    step2 = evaluator.add_parallel(
        id="U4_step2_verification",
        desc="Verify all institutional characteristics of the identified university",
        parent=uni_node,
        critical=False
    )

    # Carnegie verification (R1)
    carne = evaluator.add_parallel(
        id="U4_carnegie_verify",
        desc="Verify Carnegie R1 classification",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U4_carnegie_status",
        node_desc="University holds Carnegie R1: Very High Research Activity designation (2025 classification)",
        parent=carne,
        claim=f"{u.name} is classified as Carnegie R1 (Very High Research Activity) in the 2025 Carnegie Classification.",
        urls=u.carnegie_source_urls,
        add_ins="Prefer evidence from carnegieclassifications.acenet.edu or official *.edu institutional research/factbook explicitly stating 'R1'.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U4_carnegie_source",
        node_desc="URL reference confirming R1 classification",
        parent=carne,
        claim=f"The provided page explicitly confirms that {u.name} is R1.",
        urls=u.carnegie_source_urls,
        add_ins="At least one URL should clearly state 'R1' or 'Very High Research Activity'.",
        critical=True
    )

    # Location verification (Midwest)
    loc = evaluator.add_parallel(
        id="U4_location_verify",
        desc="Verify location in Midwestern United States",
        parent=step2,
        critical=False
    )
    state_check_node = evaluator.add_leaf(
        id="U4_state_check",
        desc="State is one of: Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, Wisconsin",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim=f"The state '{u.state}' is among the Midwestern United States list: Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, Wisconsin.",
        node=state_check_node,
        additional_instruction="Allow standard two-letter USPS abbreviations (IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI)."
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U4_location_source",
        node_desc="URL reference confirming location",
        parent=loc,
        claim=f"The provided page explicitly states that {u.name} is located in the state of {u.state}.",
        urls=u.location_source_urls,
        add_ins="Look for city/state on 'About', Contact, or campus location pages.",
        critical=True
    )

    # Institutional type (Public)
    typ = evaluator.add_parallel(
        id="U4_type_verify",
        desc="Verify public institution status",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U4_public_status",
        node_desc="University is a public (state-funded) institution",
        parent=typ,
        claim=f"{u.name} is a public institution.",
        urls=u.type_source_urls,
        add_ins="Accept phrasing like 'public research university' or 'public university'.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U4_type_source",
        node_desc="URL reference confirming institutional type",
        parent=typ,
        claim=f"The provided page explicitly indicates that {u.name} is public.",
        urls=u.type_source_urls,
        add_ins="The page should clearly state 'public' or state-system membership.",
        critical=True
    )

    # Business programs (undergraduate)
    bus = evaluator.add_parallel(
        id="U4_business_verify",
        desc="Verify business program offering",
        parent=step2,
        critical=False
    )
    bus_urls = _non_empty_url_list(([u.business_programs_url] if _is_filled(u.business_programs_url) else []) + u.business_source_urls)
    await _verify_with_required_sources(
        evaluator,
        node_id="U4_business_exists",
        node_desc="Business degree programs are offered",
        parent=bus,
        claim="The university offers undergraduate business degree programs (e.g., BBA/BS in Business, undergraduate business majors).",
        urls=bus_urls,
        add_ins="Look for a College/School of Business page or undergraduate business majors listing.",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U4_business_source",
        node_desc="URL reference to business school/program page",
        parent=bus,
        claim="This page is an official business school/undergraduate business program page for the university.",
        urls=bus_urls,
        add_ins="Accept *.edu pages clearly tied to the institution and describing undergraduate business degrees/majors.",
        critical=True
    )

    # Acceptance rate < 25% (most recent admitted class)
    acc = evaluator.add_parallel(
        id="U4_acceptance_verify",
        desc="Verify acceptance rate below 25%",
        parent=step2,
        critical=False
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U4_acceptance_threshold",
        node_desc="Acceptance rate is below 25% for most recent admitted class",
        parent=acc,
        claim="The university's acceptance (admit) rate for its most recent admitted class is below 25%.",
        urls=u.acceptance_rate_source_urls,
        add_ins="Prefer CDS (Section C), institutional research/factbook/dashboards. Reputable ranking sites acceptable if official sources not provided. Treat 'admit rate' as acceptance rate. Strictly below 25.0% (24.9% passes; 25.0% does NOT).",
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        node_id="U4_acceptance_source",
        node_desc="URL reference for acceptance rate from Common Data Set, institutional research, or ranking site",
        parent=acc,
        claim="This page reports the university's acceptance/admit rate for the latest entering class or cycle.",
        urls=u.acceptance_rate_source_urls,
        add_ins="Prefer CDS/IR official pages; reputable rankings accepted. The page should clearly present an acceptance/admit rate.",
        critical=True
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
    Evaluate an answer for the 4-university criteria task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # parallel across the four universities
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

    # 1) Extract all university information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # 2) Build and run verifications for each university
    u1 = extracted.university_1 or UniversityInfo()
    u2 = extracted.university_2 or UniversityInfo()
    u3 = extracted.university_3 or UniversityInfo()
    u4 = extracted.university_4 or UniversityInfo()

    await verify_university_1(evaluator, root, u1)
    await verify_university_2(evaluator, root, u2)
    await verify_university_3(evaluator, root, u3)
    await verify_university_4(evaluator, root, u4)

    # 3) Return structured evaluation summary
    return evaluator.get_summary()