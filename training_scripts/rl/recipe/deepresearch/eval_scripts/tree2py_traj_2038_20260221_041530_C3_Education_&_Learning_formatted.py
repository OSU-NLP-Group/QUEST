import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohio_treasurer_verification"
TASK_DESCRIPTION = """
Identify the current treasurer/CFO of Columbus City Schools (Ohio's largest school district) and verify that this individual meets all Ohio state requirements for the school district treasurer position. Specifically, you must confirm and document the following: (1) Treasurer Identification: Provide the full name of the individual currently serving as treasurer/CFO of Columbus City Schools. (2) Educational Qualifications: Verify that the treasurer holds an appropriate baccalaureate degree. Ohio requires either a baccalaureate degree in business, OR a baccalaureate degree in a non-business area plus nine semester hours in accounting. (3) Required Coursework: Confirm that the treasurer has completed both of the following required courses: three semester hours in school law and three semester hours in school finance. (4) Professional Licensure: Verify that the treasurer holds a current valid Ohio School Treasurer License issued by the Ohio State Board of Education. For each verification requirement, provide supporting documentation including official sources, institutional records, or credible reference URLs that confirm the information.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TreasurerIdentification(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EducationQualification(BaseModel):
    degree_title: Optional[str] = None  # e.g., "B.S.", "BBA"
    degree_field: Optional[str] = None  # e.g., "Accounting", "Finance"
    institution: Optional[str] = None
    business_degree: Optional[bool] = None  # True if business-related baccalaureate
    non_business_degree: Optional[bool] = None  # True if non-business baccalaureate
    accounting_hours: Optional[str] = None  # e.g., "9 semester hours"
    sources: List[str] = Field(default_factory=list)


class CourseworkSchoolLaw(BaseModel):
    completed: Optional[bool] = None
    hours: Optional[str] = None  # e.g., "3 semester hours"
    course_titles: List[str] = Field(default_factory=list)
    institution: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CourseworkSchoolFinance(BaseModel):
    completed: Optional[bool] = None
    hours: Optional[str] = None  # e.g., "3 semester hours"
    course_titles: List[str] = Field(default_factory=list)
    institution: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LicensureOhioTreasurer(BaseModel):
    license_status: Optional[str] = None  # e.g., "Valid", "Active"
    license_number: Optional[str] = None
    license_valid: Optional[bool] = None
    issue_date: Optional[str] = None
    expiration_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LicensureExam(BaseModel):
    exam_passed: Optional[bool] = None
    exam_name: Optional[str] = None
    date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ForecastsSubmission(BaseModel):
    claim: Optional[str] = None  # e.g., "District submits five-year forecasts twice annually"
    sources: List[str] = Field(default_factory=list)


class HinkleReports(BaseModel):
    claim: Optional[str] = None  # e.g., "Annual financial reports filed via Hinkle System"
    sources: List[str] = Field(default_factory=list)


class BoardContract(BaseModel):
    claim: Optional[str] = None  # e.g., "Board executed written employment contract with treasurer"
    contract_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DistrictType(BaseModel):
    district_type: Optional[str] = None  # e.g., "City School District"
    sources: List[str] = Field(default_factory=list)


class TreasurerVerificationExtraction(BaseModel):
    treasurer: Optional[TreasurerIdentification] = None
    education: Optional[EducationQualification] = None
    law_course: Optional[CourseworkSchoolLaw] = None
    finance_course: Optional[CourseworkSchoolFinance] = None
    license: Optional[LicensureOhioTreasurer] = None
    exam: Optional[LicensureExam] = None
    forecasts: Optional[ForecastsSubmission] = None
    hinkle: Optional[HinkleReports] = None
    board_contract: Optional[BoardContract] = None
    district_type: Optional[DistrictType] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_treasurer_verification() -> str:
    return """
    Extract structured information from the answer to support verification of the Columbus City Schools Treasurer/CFO and all Ohio requirements. Return a JSON object with the following fields. If any information is missing, use null for singular fields and empty arrays for lists.

    Fields to extract:
    - treasurer:
        - name: Full name of the current Treasurer/CFO of Columbus City Schools.
        - sources: All source URLs in the answer that directly support the identification (prefer official district pages, board docs, or other credible sources).
    - education:
        - degree_title: The specific baccalaureate degree title (e.g., "B.S.", "BBA").
        - degree_field: The field/major (e.g., "Accounting", "Finance", "Business Administration").
        - institution: Institution awarding the degree.
        - business_degree: true if the degree is in a business area; false otherwise; null if unspecified.
        - non_business_degree: true if the degree is in a non-business area; false otherwise; null if unspecified.
        - accounting_hours: If mentioned, specify accounting coursework hours (e.g., "9 semester hours").
        - sources: URLs that document the degree and/or coursework supporting Ohio's education rule.
    - law_course:
        - completed: true if completion of school law coursework is claimed; false if explicitly denied; null if unspecified.
        - hours: If mentioned, specify hours (e.g., "3 semester hours").
        - course_titles: Any relevant course titles mentioned (array).
        - institution: The institution where coursework was completed, if stated.
        - sources: URLs supporting completion of 3 semester hours in school law.
    - finance_course:
        - completed: true if completion of school finance coursework is claimed; false if explicitly denied; null if unspecified.
        - hours: If mentioned, specify hours (e.g., "3 semester hours").
        - course_titles: Any relevant course titles mentioned (array).
        - institution: The institution where coursework was completed, if stated.
        - sources: URLs supporting completion of 3 semester hours in school finance.
    - license:
        - license_status: Current status text if provided (e.g., "Valid", "Active").
        - license_number: License number if provided.
        - license_valid: true if the license is stated as current and valid; false otherwise; null if unspecified.
        - issue_date: License issue date if provided.
        - expiration_date: License expiration date if provided.
        - sources: URLs proving a current valid Ohio School Treasurer License (prefer official Ohio licensure/ODE/ODEW portals, district HR pages, or board docs).
    - exam:
        - exam_passed: true if passing the Ohio School Treasurer licensure exam is claimed; false otherwise; null if unspecified.
        - exam_name: Name/identifier of the exam, if provided.
        - date: Date of passing or exam, if provided.
        - sources: URLs documenting the exam pass or an official statement implying it (e.g., licensure verification that requires passing the exam).
    - forecasts:
        - claim: A concise statement in the answer indicating the district submits five-year forecasts twice annually.
        - sources: URLs supporting this (e.g., ODE/ODEW references, district pages stating May and October/November filings).
    - hinkle:
        - claim: A concise statement indicating annual financial reports are filed via the Ohio Auditor of State Hinkle System.
        - sources: URLs supporting this (e.g., Auditor of State pages or district compliance pages).
    - board_contract:
        - claim: A concise statement that the school board executed a written employment contract with the treasurer (ORC 3319).
        - contract_date: If a date is mentioned, include it.
        - sources: URLs supporting the existence of the board-approved written contract (e.g., agenda/minutes/resolutions).
    - district_type:
        - district_type: The type, e.g., "City School District".
        - sources: URLs confirming Columbus City Schools is an Ohio public school district type (e.g., ODEW district profile).

    IMPORTANT:
    - Extract ONLY URLs explicitly present in the answer; include full URLs (prepend http:// if missing). Do not invent URLs.
    - If multiple URLs are provided for a field, include them all.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_sources(sources: Optional[List[str]]) -> List[str]:
    """Normalize source list: filter empties and duplicates."""
    if not sources:
        return []
    cleaned = [s.strip() for s in sources if isinstance(s, str) and s.strip()]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in cleaned:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_identification_subtree(
    evaluator: Evaluator,
    parent_node,
    ext: TreasurerVerificationExtraction,
) -> None:
    """
    Build the 'Identify_Current_Treasurer_CFO' subtree:
    - Existence check (name + sources)
    - Source-supported verification of the identification claim
    """
    treas = ext.treasurer or TreasurerIdentification()
    name = treas.name or ""
    srcs = _norm_sources(treas.sources)

    node = evaluator.add_parallel(
        id="Identify_Current_Treasurer_CFO",
        desc="Provide the full name of the individual currently serving as treasurer/CFO of Columbus City Schools, supported by an official or otherwise credible source URL (e.g., district/board documentation).",
        parent=parent_node,
        critical=True
    )

    # Existence / sources provided (critical)
    evaluator.add_custom_node(
        result=bool(name.strip()) and len(srcs) > 0,
        id="Identify_Current_Treasurer_CFO_sources_provided",
        desc="Treasurer/CFO identification provided with at least one supporting source URL",
        parent=node,
        critical=True
    )

    # Verification leaf (critical)
    verify_leaf = evaluator.add_leaf(
        id="Identify_Current_Treasurer_CFO_supported",
        desc="Treasurer/CFO identification is supported by cited sources",
        parent=node,
        critical=True
    )
    claim = f"The current Treasurer/CFO of Columbus City Schools is {name}."
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=srcs,
        additional_instruction=(
            "Confirm that at least one provided URL explicitly states this person is the current Treasurer, "
            "Chief Financial Officer, or equivalent title for Columbus City Schools. Prefer official district or board pages. "
            "Allow reasonable title variations (e.g., 'Treasurer' vs. 'CFO' or 'Chief Financial Officer and Treasurer')."
        ),
    )


async def add_requirement_node(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    desc: str,
    sources: List[str],
    claim: str,
    add_ins: str,
) -> None:
    """
    Generic requirement node builder:
    - Adds a critical parallel node under parent
    - Adds an existence check (sources provided)
    - Adds one critical verification leaf by URLs
    """
    req_node = evaluator.add_parallel(
        id=base_id,
        desc=desc,
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(sources) > 0,
        id=f"{base_id}_sources_provided",
        desc=f"{desc} - supporting source URL(s) provided",
        parent=req_node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{base_id}_supported",
        desc=desc,
        parent=req_node,
        critical=True
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=sources,
        additional_instruction=add_ins,
    )


async def add_education_rule_nodes(
    evaluator: Evaluator,
    parent_node,
    education: EducationQualification
) -> None:
    """
    Education rule verification as a critical node with two critical leaves:
    - Degree documented
    - Rule satisfied (business degree OR non-business degree + 9 accounting hours)
    """
    srcs = _norm_sources(education.sources)
    degree_title = education.degree_title or ""
    degree_field = education.degree_field or ""
    institution = education.institution or ""
    acc_hours = education.accounting_hours or ""

    edu_node = evaluator.add_parallel(
        id="Education_Baccalaureate_Rule",
        desc="Verify the treasurer meets the education rule: baccalaureate degree in business OR baccalaureate degree in a non-business area plus nine semester hours in accounting, with supporting documentation/URLs.",
        parent=parent_node,
        critical=True
    )

    # Existence check: we need at least one source
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="Education_Baccalaureate_Rule_sources_provided",
        desc="Education rule verification - supporting source URL(s) provided",
        parent=edu_node,
        critical=True
    )

    # Leaf 1: Degree documented
    leaf_degree_doc = evaluator.add_leaf(
        id="Education_Baccalaureate_Degree_Documented",
        desc="Baccalaureate degree details (title/field/institution) are documented by sources",
        parent=edu_node,
        critical=True
    )
    claim_doc = (
        f"The individual holds a baccalaureate degree"
        f"{' (' + degree_title + ')' if degree_title else ''}"
        f"{' in ' + degree_field if degree_field else ''}"
        f"{' from ' + institution if institution else ''}."
    ).strip()
    await evaluator.verify(
        claim=claim_doc,
        node=leaf_degree_doc,
        sources=srcs,
        additional_instruction=(
            "Verify that at least one provided URL documents the person's baccalaureate degree "
            "(degree title, field/major, and/or institution). Accept official bios, resumes, LinkedIn, "
            "university pages, district/board biographies, or credential verification pages."
        ),
    )

    # Leaf 2: Rule satisfied
    leaf_rule = evaluator.add_leaf(
        id="Education_Baccalaureate_Rule_Satisfied",
        desc="Education requirement is satisfied (business baccalaureate OR non-business baccalaureate plus at least 9 semester hours in accounting)",
        parent=edu_node,
        critical=True
    )
    claim_rule = (
        "The individual's credentials satisfy Ohio's treasurer education requirement: "
        "either a baccalaureate degree in business OR a baccalaureate degree in a non-business area "
        f"plus {'at least ' + acc_hours if acc_hours else 'nine semester hours'} in accounting."
    )
    await evaluator.verify(
        claim=claim_rule,
        node=leaf_rule,
        sources=srcs,
        additional_instruction=(
            "Confirm that the sources show either a business baccalaureate (e.g., Accounting, Finance, Business Administration) "
            "or a non-business baccalaureate plus at least nine semester hours of accounting coursework. "
            "Evidence may include transcripts, official biographies listing courses, or licensure compliance records."
        ),
    )


async def build_requirements_subtree(
    evaluator: Evaluator,
    parent_node,
    ext: TreasurerVerificationExtraction,
) -> None:
    """
    Build the 'Verify_All_Explicit_Requirements' subtree with critical parallel children.
    """
    verify_all_node = evaluator.add_parallel(
        id="Verify_All_Explicit_Requirements",
        desc="Verify each explicitly stated requirement from the proposed question and constraints with supporting documentation (official sources, institutional records, or credible reference URLs).",
        parent=parent_node,
        critical=True
    )

    # 1) Education rule (critical node with two leaves)
    education = ext.education or EducationQualification()
    await add_education_rule_nodes(evaluator, verify_all_node, education)

    # 2) School Law coursework (3 semester hours)
    law = ext.law_course or CourseworkSchoolLaw()
    law_srcs = _norm_sources(law.sources)
    law_hours_txt = law.hours or "3 semester hours"
    law_desc = "Verify completion of three semester hours in school law, with supporting documentation/URLs."
    law_claim = f"The individual has completed at least three semester hours in school law ({law_hours_txt})."
    await add_requirement_node(
        evaluator, verify_all_node,
        base_id="Coursework_School_Law_3_Semester_Hours",
        desc=law_desc,
        sources=law_srcs,
        claim=law_claim,
        add_ins=(
            "Confirm that the provided URLs explicitly support completion of school law coursework totaling at least 3 semester hours. "
            "Accept course titles such as 'School Law' and official transcripts or licensure compliance records."
        )
    )

    # 3) School Finance coursework (3 semester hours)
    fin = ext.finance_course or CourseworkSchoolFinance()
    fin_srcs = _norm_sources(fin.sources)
    fin_hours_txt = fin.hours or "3 semester hours"
    fin_desc = "Verify completion of three semester hours in school finance, with supporting documentation/URLs."
    fin_claim = f"The individual has completed at least three semester hours in school finance ({fin_hours_txt})."
    await add_requirement_node(
        evaluator, verify_all_node,
        base_id="Coursework_School_Finance_3_Semester_Hours",
        desc=fin_desc,
        sources=fin_srcs,
        claim=fin_claim,
        add_ins=(
            "Confirm that the provided URLs explicitly support completion of school finance coursework totaling at least 3 semester hours. "
            "Accept course titles such as 'School Finance' or 'Public School Finance', and official transcripts/licensure records."
        )
    )

    # 4) Valid Ohio School Treasurer License
    lic = ext.license or LicensureOhioTreasurer()
    lic_srcs = _norm_sources(lic.sources)
    lic_desc = "Verify the treasurer holds a current valid Ohio School Treasurer License issued by the Ohio State Board of Education, with supporting documentation/URLs."
    lic_claim = "The individual holds a current valid Ohio School Treasurer License issued by the Ohio State Board of Education."
    await add_requirement_node(
        evaluator, verify_all_node,
        base_id="Licensure_Valid_Ohio_School_Treasurer_License",
        desc=lic_desc,
        sources=lic_srcs,
        claim=lic_claim,
        add_ins=(
            "Confirm validity and current status (e.g., 'Valid', 'Active') via official licensure lookup or district documentation. "
            "Prefer ODE/ODEW Educator Licensure pages or official district HR/board records."
        )
    )

    # 5) Licensure exam passed
    exam = ext.exam or LicensureExam()
    exam_srcs = _norm_sources(exam.sources)
    exam_desc = "Verify the treasurer (as license holder/candidate) has passed the state-administered Ohio School Treasurer licensure exam, with supporting documentation/URLs."
    exam_claim = "The individual has passed the Ohio School Treasurer licensure exam required by the state."
    await add_requirement_node(
        evaluator, verify_all_node,
        base_id="Licensure_Exam_Passed",
        desc=exam_desc,
        sources=exam_srcs,
        claim=exam_claim,
        add_ins=(
            "Confirm explicit evidence that the exam has been passed. "
            "Accept official exam results, licensure verification indicating exam passage as a prerequisite, or credible institutional records."
        )
    )

    # 6) District submits five-year forecasts twice annually (OAC 3301-92-04)
    fc = ext.forecasts or ForecastsSubmission()
    fc_srcs = _norm_sources(fc.sources)
    fc_desc = "Verify that the district submits five-year financial forecasts twice annually to the Ohio Department of Education and Workforce (per OAC 3301-92-04), with supporting documentation/URLs."
    fc_claim = (
        "Columbus City Schools submits five-year financial forecasts twice annually to the Ohio Department of Education and Workforce, "
        "consistent with OAC 3301-92-04."
    )
    await add_requirement_node(
        evaluator, verify_all_node,
        base_id="District_Submits_Five_Year_Forecasts_Twice_Annually",
        desc=fc_desc,
        sources=fc_srcs,
        claim=fc_claim,
        add_ins=(
            "It is sufficient if the provided sources explicitly state 'twice annually' or indicate submission months such as May and October/November. "
            "Prefer ODE/ODEW references or district finance pages acknowledging this requirement."
        )
    )

    # 7) Annual financial reports filed via Hinkle System
    hk = ext.hinkle or HinkleReports()
    hk_srcs = _norm_sources(hk.sources)
    hk_desc = "Verify that annual financial reports are filed with the Ohio Auditor of State through the Hinkle System, with supporting documentation/URLs."
    hk_claim = "Annual financial reports for Columbus City Schools are filed with the Ohio Auditor of State through the Hinkle System."
    await add_requirement_node(
        evaluator, verify_all_node,
        base_id="Treasurer_Files_Annual_Financial_Reports_Hinkle",
        desc=hk_desc,
        sources=hk_srcs,
        claim=hk_claim,
        add_ins=(
            "Confirm that sources explicitly reference the Ohio Auditor of State 'Hinkle System' for filing annual financial reports. "
            "Accept Auditor of State pages or district compliance statements."
        )
    )

    # 8) Board executed written employment contract (ORC 3319)
    bc = ext.board_contract or BoardContract()
    bc_srcs = _norm_sources(bc.sources)
    bc_desc = "Verify the school board executed a written contract of employment with the treasurer (per Ohio Revised Code Chapter 3319), with supporting documentation/URLs."
    bc_claim = "The school board executed a written employment contract with the treasurer, consistent with ORC 3319."
    await add_requirement_node(
        evaluator, verify_all_node,
        base_id="Board_Executed_Written_Employment_Contract",
        desc=bc_desc,
        sources=bc_srcs,
        claim=bc_claim,
        add_ins=(
            "Accept board agendas, minutes, resolutions, or district HR documents that explicitly reference a written employment contract "
            "with the treasurer. The presence of board approval and written contract suffices."
        )
    )

    # 9) District is an Ohio public school district type
    dt = ext.district_type or DistrictType()
    dt_srcs = _norm_sources(dt.sources)
    dt_desc = "Verify the district is an Ohio public school district (city, local, exempted village, or joint vocational district), with supporting documentation/URLs."
    dt_type_txt = dt.district_type or "City School District"
    dt_claim = f"Columbus City Schools is an Ohio {dt_type_txt.lower()}."
    await add_requirement_node(
        evaluator, verify_all_node,
        base_id="District_Is_Ohio_Public_School_District_Type",
        desc=dt_desc,
        sources=dt_srcs,
        claim=dt_claim,
        add_ins=(
            "Prefer ODE/ODEW district profile or other official state references. "
            "Allow reasonable wording variations as long as the classification as an Ohio public school district type is explicit."
        )
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
    Evaluate an answer for the Ohio District Treasurer complete verification task.
    """
    # Initialize evaluator (root is non-critical by framework; we add a critical top node)
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
        default_model=model,
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_treasurer_verification(),
        template_class=TreasurerVerificationExtraction,
        extraction_name="treasurer_verification_extraction",
    )

    # Build a critical top-level task node under root to enforce strict failure policy
    top_task_node = evaluator.add_sequential(
        id="Ohio_District_Treasurer_Complete_Verification",
        desc="Identify the current treasurer/CFO of Columbus City Schools and verify compliance with ALL requirements explicitly listed in the proposed question and constraints, providing credible supporting documentation/URLs for each verified requirement.",
        parent=root,
        critical=True
    )

    # Child 1: Identification (critical; if fails, subsequent nodes skipped due to sequential aggregation)
    await build_identification_subtree(evaluator, top_task_node, extracted)

    # Child 2: Verify all explicit requirements (critical parallel node)
    await build_requirements_subtree(evaluator, top_task_node, extracted)

    # Return final structured summary
    return evaluator.get_summary()