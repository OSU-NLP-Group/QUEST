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
TASK_ID = "va_ms_eng_d1_finAid_ncaa"
TASK_DESCRIPTION = """You are a current NCAA Division I student-athlete completing your undergraduate degree and planning to pursue a Master of Science (MS) in Engineering starting in Fall 2026. You want to continue your athletic career while attending graduate school, and you need to understand the financial aid options available.

Identify a public university located in Virginia that meets ALL of the following requirements:

1. The university must be accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC).

2. The university must have an NCAA Division I athletic program.

3. The university must offer a Master of Science (MS) in Engineering program (any engineering concentration) that includes a thesis option.

For the identified university, provide the following information:

University Identification:
- The full official name of the university
- Confirmation of its public institution status
- Confirmation of its Virginia location
- The official university website URL

Accreditation:
- Confirmation of SACSCOC accreditation
- A URL confirming the SACSCOC accreditation status

Athletic Program:
- Confirmation of NCAA Division I status
- A URL confirming NCAA Division I athletic program

Graduate Engineering Program:
- The specific name and concentration of the MS in Engineering program
- A URL to the MS in Engineering program information
- Confirmation that the program offers a thesis option
- The number of thesis credit hours required (if specified)
- The total credit hours required for the MS degree with thesis option
- A URL documenting the thesis option and requirements

Graduate Admission Requirements:
- The minimum GPA required for graduate admission to the MS in Engineering program
- The application deadline for Fall 2026 semester admission (domestic students)
- A URL to the graduate admission requirements

Federal Financial Aid Information:
- The federal FAFSA deadline for the 2025-2026 academic year
- The minimum credit hours required for graduate students to be eligible for federal Direct Loans
- The annual federal Direct Unsubsidized Loan limit for graduate students
- A URL to federal student aid or FAFSA deadline information

NCAA Eligibility Requirements:
- The minimum cumulative GPA required for NCAA Division I continuing athletic eligibility
- The minimum credit hours required for full-time enrollment status for NCAA Division I undergraduate student-athletes
- A URL to NCAA Division I eligibility requirements
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BasicInfo(BaseModel):
    university_name: Optional[str] = None
    is_public: Optional[str] = None
    state: Optional[str] = None
    university_website: Optional[str] = None
    public_status_url: Optional[str] = None


class AccreditationInfo(BaseModel):
    sacscoc_accredited: Optional[str] = None
    accreditation_urls: List[str] = Field(default_factory=list)


class AthleticsInfo(BaseModel):
    ncaa_division_status: Optional[str] = None
    athletics_urls: List[str] = Field(default_factory=list)


class GradProgramInfo(BaseModel):
    program_name: Optional[str] = None
    concentration: Optional[str] = None
    program_url: Optional[str] = None
    thesis_option: Optional[str] = None
    thesis_credit_hours: Optional[str] = None
    total_credit_hours_thesis: Optional[str] = None
    thesis_requirements_url: Optional[str] = None


class AdmissionInfo(BaseModel):
    min_gpa: Optional[str] = None
    fall_2026_deadline_domestic: Optional[str] = None
    admission_url: Optional[str] = None


class FederalAidInfo(BaseModel):
    fafsa_deadline_2025_2026: Optional[str] = None
    min_credits_for_loans_grad: Optional[str] = None
    annual_unsub_loan_limit_grad: Optional[str] = None
    federal_aid_urls: List[str] = Field(default_factory=list)


class NCAAInfo(BaseModel):
    min_cum_gpa_continuing: Optional[str] = None
    full_time_credits_undergrad: Optional[str] = None
    ncaa_urls: List[str] = Field(default_factory=list)


class UniversitySelection(BaseModel):
    basic: BasicInfo = Field(default_factory=BasicInfo)
    accreditation: AccreditationInfo = Field(default_factory=AccreditationInfo)
    athletics: AthleticsInfo = Field(default_factory=AthleticsInfo)
    grad_program: GradProgramInfo = Field(default_factory=GradProgramInfo)
    admission: AdmissionInfo = Field(default_factory=AdmissionInfo)
    federal_aid: FederalAidInfo = Field(default_factory=FederalAidInfo)
    ncaa: NCAAInfo = Field(default_factory=NCAAInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_selection() -> str:
    return """
    Extract the identified Virginia public university and all requested details exactly as stated in the answer. Return null for any missing field. Provide URLs only if explicitly cited in the answer.

    Required JSON structure and fields:

    {
      "basic": {
        "university_name": string|null,
        "is_public": string|null,
        "state": string|null,
        "university_website": string|null,
        "public_status_url": string|null
      },
      "accreditation": {
        "sacscoc_accredited": string|null,
        "accreditation_urls": [string, ...]
      },
      "athletics": {
        "ncaa_division_status": string|null,
        "athletics_urls": [string, ...]
      },
      "grad_program": {
        "program_name": string|null,
        "concentration": string|null,
        "program_url": string|null,
        "thesis_option": string|null,
        "thesis_credit_hours": string|null,
        "total_credit_hours_thesis": string|null,
        "thesis_requirements_url": string|null
      },
      "admission": {
        "min_gpa": string|null,
        "fall_2026_deadline_domestic": string|null,
        "admission_url": string|null
      },
      "federal_aid": {
        "fafsa_deadline_2025_2026": string|null,
        "min_credits_for_loans_grad": string|null,
        "annual_unsub_loan_limit_grad": string|null,
        "federal_aid_urls": [string, ...]
      },
      "ncaa": {
        "min_cum_gpa_continuing": string|null,
        "full_time_credits_undergrad": string|null,
        "ncaa_urls": [string, ...]
      }
    }

    Rules:
    - "is_public" should be a short confirmation phrase like "public", "public university", or similar (do not infer if not stated).
    - "state" should be the U.S. state name; if Virginia is stated, capture "Virginia".
    - For URL fields, include only valid, full URLs explicitly present in the answer text.
    - For arrays of URLs, include all distinct URLs cited; if none are cited, return an empty array.
    - For numeric details like credit hours or GPA or loan amounts, keep them as strings exactly as written (e.g., "30 credits", "2.7 GPA", "$20,500").
    - Do not invent or normalize information; extract literally from the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _safe_sources(*urls_or_lists: Optional[Any]) -> List[str]:
    """Collect strings and list elements into a flat list of non-empty URLs (unique order-preserving)."""
    out: List[str] = []
    seen = set()
    for item in urls_or_lists:
        if not item:
            continue
        if isinstance(item, str):
            cand = item.strip()
            if cand and cand not in seen:
                out.append(cand)
                seen.add(cand)
        elif isinstance(item, list):
            for u in item:
                if isinstance(u, str):
                    cand = u.strip()
                    if cand and cand not in seen:
                        out.append(cand)
                        seen.add(cand)
    return out


# --------------------------------------------------------------------------- #
# Verification group builders                                                 #
# --------------------------------------------------------------------------- #
async def verify_university_identification(evaluator: Evaluator, parent, data: UniversitySelection):
    node = evaluator.add_parallel(
        id="University_Identification_and_Location",
        desc="Verify the university's basic identification, public status, and Virginia location",
        parent=parent,
        critical=False
    )
    uni = data.basic
    uni_name = uni.university_name or "the university"
    website = uni.university_website

    # Existence of official website URL
    evaluator.add_custom_node(
        result=_non_empty(website),
        id="University_Website_Reference",
        desc="Provide the official university website URL",
        parent=node,
        critical=True
    )

    # University name verification
    name_leaf = evaluator.add_leaf(
        id="University_Name",
        desc="Provide the full official name of the university",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The full official name of the university is '{uni.university_name}'.",
        node=name_leaf,
        sources=website,
        additional_instruction="Use the university's official homepage or About page to confirm the formal official name. Minor formatting differences (e.g., '&' vs 'and') are acceptable."
    )

    # Public institution verification
    public_leaf = evaluator.add_leaf(
        id="Public_Institution",
        desc="Verify the university is a public institution",
        parent=node,
        critical=True
    )
    pub_sources = _safe_sources(uni.public_status_url, website)
    await evaluator.verify(
        claim=f"'{uni_name}' is a public university (public institution).",
        node=public_leaf,
        sources=pub_sources,
        additional_instruction="Confirm that the institution is publicly funded or identified as a 'public university' or 'public institution' on an official page (About, Facts, governance)."
    )

    # Virginia location verification
    va_leaf = evaluator.add_leaf(
        id="Virginia_Location",
        desc="Verify the university is located in Virginia",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{uni_name}' is located in the U.S. state of Virginia.",
        node=va_leaf,
        sources=website,
        additional_instruction="Verify the institution's location is in the state of Virginia using the official site (contact, address, About page)."
    )


async def verify_accreditation(evaluator: Evaluator, parent, data: UniversitySelection):
    node = evaluator.add_parallel(
        id="Accreditation_Verification",
        desc="Verify the university's regional accreditation status",
        parent=parent,
        critical=False
    )

    urls = data.accreditation.accreditation_urls
    uni_name = data.basic.university_name or "the university"

    # Existence of accreditation URL
    evaluator.add_custom_node(
        result=len(_safe_sources(urls)) > 0,
        id="Accreditation_Reference_URL",
        desc="Provide a URL confirming SACSCOC accreditation status",
        parent=node,
        critical=True
    )

    # SACSCOC accreditation verification
    sacscoc_leaf = evaluator.add_leaf(
        id="SACSCOC_Accreditation",
        desc="Verify the university is accredited by SACSCOC",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{uni_name}' is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC).",
        node=sacscoc_leaf,
        sources=urls,
        additional_instruction="Look for explicit mentions of 'SACSCOC' or 'Southern Association of Colleges and Schools Commission on Colleges' on either the SACSCOC directory or the university's accreditation page."
    )


async def verify_athletics(evaluator: Evaluator, parent, data: UniversitySelection):
    node = evaluator.add_parallel(
        id="Athletic_Program_Verification",
        desc="Verify the university's NCAA Division I athletic program",
        parent=parent,
        critical=False
    )

    urls = data.athletics.athletics_urls
    uni_name = data.basic.university_name or "the university"

    # Existence of athletics/confirmation URL
    evaluator.add_custom_node(
        result=len(_safe_sources(urls)) > 0,
        id="Athletic_Program_Reference_URL",
        desc="Provide a URL confirming NCAA Division I athletic status",
        parent=node,
        critical=True
    )

    # Division I verification
    d1_leaf = evaluator.add_leaf(
        id="NCAA_Division_I_Status",
        desc="Verify the university has an NCAA Division I athletic program",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{uni_name}' competes in NCAA Division I athletics.",
        node=d1_leaf,
        sources=urls,
        additional_instruction="Accept 'NCAA Division I', 'NCAA DI', 'D-I' language. Confirmation may appear on NCAA.org, conference sites, or the school's official athletics site."
    )


async def verify_grad_program(evaluator: Evaluator, parent, data: UniversitySelection):
    gp_root = evaluator.add_sequential(
        id="Graduate_Engineering_Program",
        desc="Verify the university offers a suitable MS in Engineering program with thesis option",
        parent=parent,
        critical=False
    )

    gp = data.grad_program
    uni_name = data.basic.university_name or "the university"

    # Step 1: Program exists (parallel)
    prog_node = evaluator.add_parallel(
        id="MS_Engineering_Program_Exists",
        desc="Verify the university offers an MS in Engineering program (any concentration)",
        parent=gp_root,
        critical=False
    )

    # Program URL existence
    evaluator.add_custom_node(
        result=_non_empty(gp.program_url),
        id="Program_Reference_URL",
        desc="Provide a URL to the MS in Engineering program information",
        parent=prog_node,
        critical=True
    )

    # Program name/type verification
    prog_leaf = evaluator.add_leaf(
        id="Program_Name_and_Type",
        desc="Provide the specific MS in Engineering program name and concentration",
        parent=prog_node,
        critical=True
    )
    prog_claim = (
        f"This page describes a Master of Science (MS) graduate program in engineering or an engineering discipline. "
        f"The program name is '{gp.program_name}'."
    )
    await evaluator.verify(
        claim=prog_claim,
        node=prog_leaf,
        sources=gp.program_url,
        additional_instruction="It counts if the page is for an MS in any engineering field (e.g., MS in Mechanical Engineering, Electrical Engineering, or 'MS in Engineering'). Accept common abbreviations (MS, M.S.)."
    )

    # Step 2: Thesis option available (parallel)
    thesis_node = evaluator.add_parallel(
        id="Thesis_Option_Available",
        desc="Verify the MS program offers a thesis option",
        parent=gp_root,
        critical=False
    )

    thesis_url_pref = gp.thesis_requirements_url if _non_empty(gp.thesis_requirements_url) else gp.program_url

    # Thesis option URL existence (prefer dedicated requirements/handbook if provided; fallback to program page)
    evaluator.add_custom_node(
        result=_non_empty(gp.thesis_requirements_url) or _non_empty(gp.program_url),
        id="Thesis_Option_Reference_URL",
        desc="Provide a URL documenting the thesis option and requirements",
        parent=thesis_node,
        critical=True
    )

    # Thesis option confirmation
    thesis_opt_leaf = evaluator.add_leaf(
        id="Thesis_Option_Confirmation",
        desc="Confirm the program explicitly offers a thesis track or option",
        parent=thesis_node,
        critical=True
    )
    await evaluator.verify(
        claim="This program offers a thesis option (thesis track) for the MS degree.",
        node=thesis_opt_leaf,
        sources=thesis_url_pref,
        additional_instruction="Look for explicit 'thesis option', 'thesis track', or similar in degree requirements/handbook/catalog."
    )

    # Thesis credit hours (non-critical)
    if _non_empty(gp.thesis_credit_hours):
        thesis_credits_leaf = evaluator.add_leaf(
            id="Thesis_Credit_Hours",
            desc="Provide the number of thesis credit hours required (if specified)",
            parent=thesis_node,
            critical=False
        )
        await evaluator.verify(
            claim=f"The thesis component requires {gp.thesis_credit_hours} of thesis credit hours (or equivalent independent research credits).",
            node=thesis_credits_leaf,
            sources=thesis_url_pref,
            additional_instruction="Verify the stated number of thesis or research credits in the degree requirements."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Thesis_Credit_Hours",
            desc="Provide the number of thesis credit hours required (if specified)",
            parent=thesis_node,
            critical=False
        )

    # Total program credit hours with thesis (critical)
    if _non_empty(gp.total_credit_hours_thesis):
        total_credits_leaf = evaluator.add_leaf(
            id="Total_Program_Credit_Hours",
            desc="Provide the total credit hours required for the MS degree with thesis option",
            parent=thesis_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The total credits required for the MS degree with the thesis option are {gp.total_credit_hours_thesis}.",
            node=total_credits_leaf,
            sources=thesis_url_pref,
            additional_instruction="Confirm the total minimum credits for the MS with the thesis option from the official program/handbook/catalog page."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Total_Program_Credit_Hours",
            desc="Provide the total credit hours required for the MS degree with thesis option",
            parent=thesis_node,
            critical=True
        )

    # Step 3: Graduate admission requirements (parallel)
    adm_node = evaluator.add_parallel(
        id="Graduate_Admission_Requirements",
        desc="Provide information about graduate admission requirements",
        parent=gp_root,
        critical=False
    )

    # Admission URL existence
    evaluator.add_custom_node(
        result=_non_empty(data.admission.admission_url),
        id="Admission_Requirements_Reference_URL",
        desc="Provide a URL to graduate admission requirements",
        parent=adm_node,
        critical=True
    )

    # Minimum GPA
    if _non_empty(data.admission.min_gpa):
        min_gpa_leaf = evaluator.add_leaf(
            id="Minimum_GPA_Requirement",
            desc="Provide the minimum GPA required for graduate admission",
            parent=adm_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The minimum GPA required for admission to the MS Engineering program is {data.admission.min_gpa}.",
            node=min_gpa_leaf,
            sources=data.admission.admission_url,
            additional_instruction="Verify the stated minimum undergraduate GPA for admission; accept reasonable wording variants (overall GPA, cumulative GPA)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Minimum_GPA_Requirement",
            desc="Provide the minimum GPA required for graduate admission",
            parent=adm_node,
            critical=True
        )

    # Fall 2026 application deadline (domestic)
    if _non_empty(data.admission.fall_2026_deadline_domestic):
        deadline_leaf = evaluator.add_leaf(
            id="Application_Deadline_Fall",
            desc="Provide the application deadline for Fall semester admission",
            parent=adm_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The application deadline for domestic applicants for Fall 2026 admission is {data.admission.fall_2026_deadline_domestic}.",
            node=deadline_leaf,
            sources=data.admission.admission_url,
            additional_instruction="Confirm the date for Fall 2026 (domestic). If multiple rounds (priority/final) are listed, the stated one must appear on the page."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Application_Deadline_Fall",
            desc="Provide the application deadline for Fall semester admission",
            parent=adm_node,
            critical=True
        )


async def verify_financial_aid(evaluator: Evaluator, parent, data: UniversitySelection):
    node = evaluator.add_parallel(
        id="Financial_Aid_Information",
        desc="Provide federal financial aid eligibility and deadline information",
        parent=parent,
        critical=False
    )

    fed_urls = data.federal_aid.federal_aid_urls

    # Existence of federal aid/FAFSA URL(s)
    evaluator.add_custom_node(
        result=len(_safe_sources(fed_urls)) > 0,
        id="Financial_Aid_Reference_URL",
        desc="Provide a URL to federal student aid or FAFSA information",
        parent=node,
        critical=True
    )

    # FAFSA federal deadline 2025-2026
    if _non_empty(data.federal_aid.fafsa_deadline_2025_2026):
        fafsa_leaf = evaluator.add_leaf(
            id="FAFSA_Deadline_2025_2026",
            desc="Provide the federal FAFSA deadline for the 2025-2026 academic year",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The federal FAFSA deadline for the 2025–2026 academic year is {data.federal_aid.fafsa_deadline_2025_2026}.",
            node=fafsa_leaf,
            sources=fed_urls,
            additional_instruction="Verify using an official federal source (studentaid.gov or FAFSA). Minor formatting differences in the date are acceptable if equivalent."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="FAFSA_Deadline_2025_2026",
            desc="Provide the federal FAFSA deadline for the 2025-2026 academic year",
            parent=node,
            critical=True
        )

    # Minimum enrollment for graduate students to be eligible for Direct Loans
    if _non_empty(data.federal_aid.min_credits_for_loans_grad):
        min_enroll_leaf = evaluator.add_leaf(
            id="Graduate_Student_Minimum_Enrollment",
            desc="Provide the minimum credit hours required for graduate students to be eligible for federal loans",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"Graduate students must be enrolled at least {data.federal_aid.min_credits_for_loans_grad} (or at least half-time as defined) to be eligible for federal Direct Loans.",
            node=min_enroll_leaf,
            sources=fed_urls,
            additional_instruction="Allow formulations that specify 'at least half-time' without a numeric credit-hour; if a number is provided, verify it matches the referenced policy."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Graduate_Student_Minimum_Enrollment",
            desc="Provide the minimum credit hours required for graduate students to be eligible for federal loans",
            parent=node,
            critical=True
        )

    # Annual federal Direct Unsubsidized Loan limit for graduate students
    if _non_empty(data.federal_aid.annual_unsub_loan_limit_grad):
        loan_limit_leaf = evaluator.add_leaf(
            id="Graduate_Loan_Annual_Limit",
            desc="Provide the annual federal Direct Unsubsidized Loan limit for graduate students",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The annual federal Direct Unsubsidized Loan limit for graduate students is {data.federal_aid.annual_unsub_loan_limit_grad}.",
            node=loan_limit_leaf,
            sources=fed_urls,
            additional_instruction="Verify using an official federal source (studentaid.gov). Accept reasonable formatting like '$20,500' or '20,500 USD'."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Graduate_Loan_Annual_Limit",
            desc="Provide the annual federal Direct Unsubsidized Loan limit for graduate students",
            parent=node,
            critical=True
        )


async def verify_ncaa_eligibility(evaluator: Evaluator, parent, data: UniversitySelection):
    node = evaluator.add_parallel(
        id="NCAA_Eligibility_Requirements",
        desc="Provide NCAA Division I continuing eligibility requirements for student-athletes",
        parent=parent,
        critical=False
    )

    ncaa_urls = data.ncaa.ncaa_urls

    # Existence of NCAA eligibility URL
    evaluator.add_custom_node(
        result=len(_safe_sources(ncaa_urls)) > 0,
        id="NCAA_Eligibility_Reference_URL",
        desc="Provide a URL to NCAA eligibility requirements",
        parent=node,
        critical=True
    )

    # Minimum cumulative GPA for continuing eligibility
    if _non_empty(data.ncaa.min_cum_gpa_continuing):
        gpa_leaf = evaluator.add_leaf(
            id="Minimum_GPA_Continuing_Eligibility",
            desc="Provide the minimum cumulative GPA required for NCAA Division I continuing athletic eligibility",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The minimum cumulative GPA required for NCAA Division I continuing athletic eligibility is {data.ncaa.min_cum_gpa_continuing}.",
            node=gpa_leaf,
            sources=ncaa_urls,
            additional_instruction="Verify from NCAA Division I bylaws/guide or official NCAA pages. Accept policy statements equivalent in meaning."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Minimum_GPA_Continuing_Eligibility",
            desc="Provide the minimum cumulative GPA required for NCAA Division I continuing athletic eligibility",
            parent=node,
            critical=True
        )

    # Full-time enrollment credits for undergraduate student-athletes
    if _non_empty(data.ncaa.full_time_credits_undergrad):
        ft_leaf = evaluator.add_leaf(
            id="Full_Time_Enrollment_Requirement",
            desc="Provide the minimum credit hours required for full-time enrollment status for NCAA Division I undergraduate student-athletes",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"For NCAA Division I undergraduate student-athletes, full-time enrollment requires at least {data.ncaa.full_time_credits_undergrad} credit hours in a term (or the institution's defined full-time threshold).",
            node=ft_leaf,
            sources=ncaa_urls,
            additional_instruction="Commonly 12 credits is full-time; verify against NCAA or institutional compliance pages that reference NCAA rules."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Full_Time_Enrollment_Requirement",
            desc="Provide the minimum credit hours required for full-time enrollment status for NCAA Division I undergraduate student-athletes",
            parent=node,
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator (root is a non-critical parallel aggregator to allow partial credit aggregation)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_selection(),
        template_class=UniversitySelection,
        extraction_name="university_selection",
    )

    # Build verification tree according to rubric
    # Group 1: University Identification and Location
    await verify_university_identification(evaluator, root, extracted)

    # Group 2: Accreditation Verification
    await verify_accreditation(evaluator, root, extracted)

    # Group 3: Athletic Program Verification
    await verify_athletics(evaluator, root, extracted)

    # Group 4: Graduate Engineering Program (sequential sub-steps)
    await verify_grad_program(evaluator, root, extracted)

    # Group 5: Federal Financial Aid Information
    await verify_financial_aid(evaluator, root, extracted)

    # Group 6: NCAA Eligibility Requirements
    await verify_ncaa_eligibility(evaluator, root, extracted)

    # Return the final structured summary
    return evaluator.get_summary()