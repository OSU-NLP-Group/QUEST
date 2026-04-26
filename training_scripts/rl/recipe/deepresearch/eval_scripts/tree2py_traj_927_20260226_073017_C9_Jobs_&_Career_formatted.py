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
TASK_ID = "us_universities_career_preparation"
TASK_DESCRIPTION = (
    "Identify four universities in the United States that demonstrate comprehensive career preparation and support programs. "
    "For each university, provide the following information:\n"
    "1. Basic Information: The official university name, its location (U.S. state), and confirmation it is located in the United States.\n"
    "2. Mandatory Experiential Learning Requirement: Documentation that the university has a mandatory experiential learning, "
    "cooperative education (co-op), or internship requirement for graduation. Specify the type of requirement (credits, hours, or number of experiences), "
    "the quantitative requirement, and which student populations it applies to.\n"
    "3. Career Services Structure: The official name of the career services office or center, the administrative unit it reports to (if available), "
    "and its physical location or building.\n"
    "4. Career Services Offerings: Confirmation that the career services office provides all of the following services: "
    "individual career counseling; resume and cover letter assistance; interview preparation or mock interviews; in-person career fairs; "
    "internship search and placement assistance; documented employer partnership programs.\n"
    "5. Graduate Outcomes Data: Documentation that the university publicly reports post-graduation career outcomes data, including the employment rate or percentage of graduates employed, "
    "the timeframe for outcomes measurement (e.g., 6 months after graduation), and confirmation that the data is from a recent graduating class (within the last 3 years, "
    "meaning Class of 2022 or later as of February 2026).\n"
    "For all information provided, include the specific URLs where each piece of information can be verified."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniBasic(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None  # e.g., "United States"
    urls: List[str] = Field(default_factory=list)  # Sources for basic info


class ExperientialReq(BaseModel):
    requirement_exists: Optional[str] = None  # "yes"/"no" if mentioned
    requirement_type: Optional[str] = None  # credits/hours/experiences
    requirement_quantity: Optional[str] = None  # e.g., "2 co-op rotations", "120 hours", "3 credits"
    applicable_students: Optional[str] = None  # e.g., "all undergraduates", "engineering majors"
    urls: List[str] = Field(default_factory=list)  # Sources for experiential program


class CareerStructure(BaseModel):
    office_name: Optional[str] = None  # e.g., "Career Services Center", "Center for Career Development"
    reports_to: Optional[str] = None  # e.g., "Student Affairs", "Provost"
    location: Optional[str] = None  # e.g., building name or address
    urls: List[str] = Field(default_factory=list)  # Sources for structure details


class CareerOfferings(BaseModel):
    counseling: Optional[str] = None  # presence indicator text from answer
    resume_services: Optional[str] = None
    interview_prep: Optional[str] = None
    career_fairs_in_person: Optional[str] = None
    internship_assistance: Optional[str] = None
    employer_partnerships: Optional[str] = None
    urls: List[str] = Field(default_factory=list)  # Sources for offerings


class Outcomes(BaseModel):
    reports_public: Optional[str] = None  # indicator text that outcomes are public
    employment_rate: Optional[str] = None  # e.g., "92%" or "92 percent employed"
    timeframe: Optional[str] = None  # e.g., "6 months after graduation"
    recent_class: Optional[str] = None  # e.g., "Class of 2024"
    urls: List[str] = Field(default_factory=list)  # Sources for outcomes


class UniversityInfo(BaseModel):
    basic: Optional[UniBasic] = None
    experiential: Optional[ExperientialReq] = None
    structure: Optional[CareerStructure] = None
    offerings: Optional[CareerOfferings] = None
    outcomes: Optional[Outcomes] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return (
        "Extract up to four universities and their details from the answer. For each university, return a structured object with the following fields:\n"
        "- basic: { name, state, country, urls }\n"
        "    • name: official university name as written in the answer; if multiple variants are provided, choose the most official form.\n"
        "    • state: U.S. state as written in the answer (e.g., 'Massachusetts', 'CA'; use the form in the answer).\n"
        "    • country: country as written in the answer (e.g., 'United States', 'USA'); if not explicitly stated, use null.\n"
        "    • urls: list of URLs in the answer that substantiate the basic info (official site, about page, contact page, etc.).\n"
        "- experiential: { requirement_exists, requirement_type, requirement_quantity, applicable_students, urls }\n"
        "    • requirement_exists: 'yes' or 'no' based on the answer's wording about a mandatory experiential learning/co-op/internship graduation requirement.\n"
        "    • requirement_type: one of {credits, hours, experiences} or a descriptive text if ambiguous; extract exactly as in the answer.\n"
        "    • requirement_quantity: the quantitative requirement as written (e.g., '2 co-op rotations', '120 hours', '3 credits').\n"
        "    • applicable_students: which student populations this applies to as written (e.g., 'all undergraduates', 'engineering majors').\n"
        "    • urls: URLs in the answer that substantiate the requirement, preferably official catalog/policy/program pages.\n"
        "- structure: { office_name, reports_to, location, urls }\n"
        "    • office_name: official name of the career services office/center.\n"
        "    • reports_to: administrative unit it reports to (if mentioned).\n"
        "    • location: physical location/building (if mentioned).\n"
        "    • urls: URLs that substantiate structure info (career services website, org chart, contact page).\n"
        "- offerings: { counseling, resume_services, interview_prep, career_fairs_in_person, internship_assistance, employer_partnerships, urls }\n"
        "    • Each field should be a short confirmation/excerpt as presented in the answer (or null if not provided).\n"
        "    • urls: URLs supporting offerings (service pages, events pages, employer partnerships page).\n"
        "- outcomes: { reports_public, employment_rate, timeframe, recent_class, urls }\n"
        "    • reports_public: short confirmation/excerpt indicating public reporting exists.\n"
        "    • employment_rate: the employment/placement rate percentage as written.\n"
        "    • timeframe: the measurement timeframe as written (e.g., '6 months after graduation').\n"
        "    • recent_class: class year mentioned (e.g., 'Class of 2023').\n"
        "    • urls: URLs to the outcomes report/dashboard.\n"
        "Ensure:\n"
        "- Extract only what is explicitly in the answer. Do not invent values.\n"
        "- Extract all URLs precisely as they appear; include full URLs. If URLs are in markdown, return the actual URL.\n"
        "- If any field is missing in the answer, set it to null (or empty list for urls).\n"
        "- Return a JSON object with a 'universities' array of these per-university objects."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(v: Optional[str]) -> str:
    return v or ""


def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_basic_information(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    idx: int
) -> None:
    basic_node = evaluator.add_parallel(
        id=f"university_{idx+1}_basic_information",
        desc="Basic university identification and location information",
        parent=parent_node,
        critical=False  # adjusted to allow partial credit and non-critical children
    )

    basic = uni.basic or UniBasic()

    # 1) Official university name
    name_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_university_name",
        desc="Provides the official name of the university",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the university is '{_safe(basic.name)}'.",
        node=name_leaf,
        sources=_urls_or_empty(basic.urls),
        additional_instruction=(
            "Verify the institution's official name on the provided page(s). Allow minor variations such as 'University of X' vs 'X University'."
        )
    )

    # 2) Confirms located in US
    us_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_us_location",
        desc="Confirms the university is located in the United States",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim="The university is located in the United States.",
        node=us_leaf,
        sources=_urls_or_empty(basic.urls),
        additional_instruction=(
            "Confirm U.S. location via address, 'USA', or other explicit indicators on the provided page(s)."
        )
    )

    # 3) State identification
    state_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_state_identification",
        desc="Identifies the specific U.S. state where the university is located",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The university is located in the state of '{_safe(basic.state)}'.",
        node=state_leaf,
        sources=_urls_or_empty(basic.urls),
        additional_instruction=(
            "Check the campus address or location page to verify the state. Allow full state names or USPS abbreviations."
        )
    )

    # 4) Basic info sources presence
    sources_exists = evaluator.add_custom_node(
        result=len(_urls_or_empty(basic.urls)) > 0,
        id=f"university_{idx+1}_basic_info_sources",
        desc="Provides URL references for basic university information",
        parent=basic_node,
        critical=True
    )


async def verify_experiential_learning(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    idx: int
) -> None:
    exp_node = evaluator.add_sequential(
        id=f"university_{idx+1}_experiential_learning_program",
        desc="Mandatory experiential learning or cooperative education program requirements",
        parent=parent_node,
        critical=False  # adjusted to allow non-critical children given rubric includes non-critical 'applicable_students'
    )

    exp = uni.experiential or ExperientialReq()

    # A) Existence of mandatory requirement
    existence_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_program_existence",
        desc="Confirms the university has a mandatory experiential learning, co-op, or internship requirement for graduation",
        parent=exp_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The university has a mandatory experiential learning, co-op, or internship requirement for graduation."
        ),
        node=existence_leaf,
        sources=_urls_or_empty(exp.urls),
        additional_instruction=(
            "Look for explicit language indicating 'required' for graduation or degree completion, on official policy/catalog/program pages."
        )
    )

    # B) Detailed requirements (parallel)
    reqs_node = evaluator.add_parallel(
        id=f"university_{idx+1}_program_requirements",
        desc="Details of the experiential learning program requirements",
        parent=exp_node,
        critical=False  # adjusted due to some non-critical child (applicable_students)
    )

    # Type of requirement
    req_type_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_requirement_type",
        desc="Specifies the type of requirement (credits, hours, or number of experiences)",
        parent=reqs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The requirement type is '{_safe(exp.requirement_type)}' (credits, hours, or number of experiences).",
        node=req_type_leaf,
        sources=_urls_or_empty(exp.urls),
        additional_instruction=(
            "Allow synonyms (e.g., 'units' for credits, 'terms' or 'rotations' for experiences) if clearly equivalent."
        )
    )

    # Quantity
    req_qty_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_requirement_quantity",
        desc="Specifies the quantitative requirement (e.g., number of credits, hours, or experiences)",
        parent=reqs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The quantitative requirement is '{_safe(exp.requirement_quantity)}'.",
        node=req_qty_leaf,
        sources=_urls_or_empty(exp.urls),
        additional_instruction=(
            "Verify the stated number of credits/hours/experiences as written on the official page(s)."
        )
    )

    # Applicable student populations (non-critical)
    applies_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_applicable_students",
        desc="Specifies which student populations the requirement applies to (e.g., all undergraduates, specific colleges)",
        parent=reqs_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The requirement applies to '{_safe(exp.applicable_students)}'.",
        node=applies_leaf,
        sources=_urls_or_empty(exp.urls),
        additional_instruction=(
            "Check if the page indicates which students must complete the requirement (e.g., 'all undergraduates' or specific colleges/majors)."
        )
    )

    # Sources presence
    prog_sources = evaluator.add_custom_node(
        result=len(_urls_or_empty(exp.urls)) > 0,
        id=f"university_{idx+1}_program_sources",
        desc="Provides URL references for experiential learning program information",
        parent=reqs_node,
        critical=True
    )


async def verify_career_structure(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    idx: int
) -> None:
    struct_node = evaluator.add_parallel(
        id=f"university_{idx+1}_career_services_structure",
        desc="Career services office organizational structure and staffing",
        parent=parent_node,
        critical=False  # adjusted to allow non-critical children as per rubric for reporting/location
    )

    struct = uni.structure or CareerStructure()

    # Office name
    office_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_office_name",
        desc="Identifies the official name of the career services office or center",
        parent=struct_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official career services office/center name is '{_safe(struct.office_name)}'.",
        node=office_leaf,
        sources=_urls_or_empty(struct.urls),
        additional_instruction=(
            "Verify the official office name on the career services website or university directory; allow minor naming variations."
        )
    )

    # Reporting structure (non-critical)
    report_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_reporting_structure",
        desc="Identifies the administrative unit the career services office reports to (e.g., Student Affairs, Academic Affairs)",
        parent=struct_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The career services office reports to '{_safe(struct.reports_to)}'.",
        node=report_leaf,
        sources=_urls_or_empty(struct.urls),
        additional_instruction=(
            "Confirm reporting line if available via org charts, about pages, or official descriptions."
        )
    )

    # Physical location (non-critical)
    loc_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_physical_location",
        desc="Provides the physical location or building where career services is housed",
        parent=struct_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The career services office is housed at '{_safe(struct.location)}'.",
        node=loc_leaf,
        sources=_urls_or_empty(struct.urls),
        additional_instruction=(
            "Verify building or address via contact/location pages; allow reasonable synonyms for building names."
        )
    )

    # Sources presence
    struct_sources = evaluator.add_custom_node(
        result=len(_urls_or_empty(struct.urls)) > 0,
        id=f"university_{idx+1}_structure_sources",
        desc="Provides URL references for career services structure information",
        parent=struct_node,
        critical=True
    )


async def verify_career_offerings(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    idx: int
) -> None:
    offer_node = evaluator.add_parallel(
        id=f"university_{idx+1}_career_services_offerings",
        desc="Types of career services and programs offered",
        parent=parent_node,
        critical=False  # parent non-critical with critical children leaves
    )

    off = uni.offerings or CareerOfferings()
    srcs = _urls_or_empty(off.urls)

    # Helper to create and verify a service leaf
    async def _verify_service(service_id: str, desc: str, claim_text: str, add_ins: str):
        leaf = evaluator.add_leaf(
            id=f"university_{idx+1}_{service_id}",
            desc=desc,
            parent=offer_node,
            critical=True
        )
        await evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=srcs,
            additional_instruction=add_ins
        )

    await _verify_service(
        "career_counseling",
        "Confirms the university offers individual career counseling services",
        "The career services office provides individual career counseling or one-on-one advising.",
        "Verify via service descriptions or appointment pages; allow synonyms like 'career coaching' or 'advising'."
    )

    await _verify_service(
        "resume_services",
        "Confirms the university offers resume and cover letter assistance",
        "The career services office offers resume and cover letter assistance.",
        "Look for workshops, drop-ins, templates, or individualized assistance pages."
    )

    await _verify_service(
        "interview_preparation",
        "Confirms the university offers interview preparation or mock interviews",
        "The career services office provides interview preparation or mock interviews.",
        "Check service pages mentioning mock interviews, interview prep, or practice resources."
    )

    await _verify_service(
        "in_person_career_fairs",
        "Confirms the university conducts in-person career fairs",
        "The university conducts in-person or on-campus career fairs.",
        "Verify via events listings or employer fair pages; in-person language or campus venue indicates in-person."
    )

    await _verify_service(
        "internship_assistance",
        "Confirms the university provides internship search and placement assistance",
        "The career services office provides internship search and placement assistance.",
        "Look for internship search help, experiential learning support, or placement assistance descriptions."
    )

    await _verify_service(
        "employer_partnerships",
        "Confirms the university has documented employer partnership programs",
        "The university has documented employer partnership programs.",
        "Verify via employer partner listings, partnership program pages, or recruiting information."
    )

    # Sources presence
    offer_sources = evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"university_{idx+1}_services_sources",
        desc="Provides URL references for career services offerings",
        parent=offer_node,
        critical=True
    )


async def verify_graduate_outcomes(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    idx: int
) -> None:
    out_node = evaluator.add_sequential(
        id=f"university_{idx+1}_graduate_outcomes",
        desc="Post-graduation career outcomes data and reporting",
        parent=parent_node,
        critical=False  # adjusted to allow non-critical child parallel node
    )

    out = uni.outcomes or Outcomes()
    srcs = _urls_or_empty(out.urls)

    # A) Reporting exists
    report_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_outcomes_reporting",
        desc="Confirms the university publicly reports post-graduation career outcomes data",
        parent=out_node,
        critical=True
    )
    await evaluator.verify(
        claim="The university publicly reports post-graduation career outcomes data.",
        node=report_leaf,
        sources=srcs,
        additional_instruction=(
            "Confirm a public outcomes report or dashboard exists on the provided page(s)."
        )
    )

    # B) Metrics (parallel)
    metrics_node = evaluator.add_parallel(
        id=f"university_{idx+1}_outcomes_metrics",
        desc="Specific metrics and data points from outcomes reporting",
        parent=out_node,
        critical=False
    )

    # Employment rate
    emp_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_employment_rate",
        desc="Reports the employment rate or percentage of graduates employed",
        parent=metrics_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The employment or placement rate reported is '{_safe(out.employment_rate)}'.",
        node=emp_leaf,
        sources=srcs,
        additional_instruction=(
            "Accept employment rate, placement rate, or employment plus continuing education rate if clearly labeled."
        )
    )

    # Timeframe
    tf_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_outcome_timeframe",
        desc="Specifies the timeframe for outcomes measurement (e.g., 6 months after graduation)",
        parent=metrics_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The outcomes measurement timeframe is '{_safe(out.timeframe)}'.",
        node=tf_leaf,
        sources=srcs,
        additional_instruction=(
            "Common timeframes include 6 months after graduation; verify as stated on the outcomes page."
        )
    )

    # Recent data (Class of 2022 or later as of Feb 2026)
    recent_leaf = evaluator.add_leaf(
        id=f"university_{idx+1}_recent_data",
        desc="Confirms the outcomes data is from a recent graduating class (within last 3 years)",
        parent=metrics_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The outcomes report references a recent graduating class such as '{_safe(out.recent_class)}', "
            "which is Class of 2022 or later (as of February 2026)."
        ),
        node=recent_leaf,
        sources=srcs,
        additional_instruction=(
            "Treat 'recent' as Class of 2022 or later (2022, 2023, 2024, or 2025). Verify the class year mentioned on the page."
        )
    )

    # Sources presence
    out_sources = evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"university_{idx+1}_outcomes_sources",
        desc="Provides URL references for graduate outcomes data",
        parent=metrics_node,
        critical=True
    )


async def verify_university(
    evaluator: Evaluator,
    root_node,
    uni: UniversityInfo,
    idx: int
) -> None:
    # University node (parallel, non-critical to allow partial credit per item)
    uni_node = evaluator.add_parallel(
        id=f"university_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying university meeting all specified criteria",
        parent=root_node,
        critical=False
    )

    # Build and verify sub-sections
    await verify_basic_information(evaluator, uni_node, uni, idx)
    await verify_experiential_learning(evaluator, uni_node, uni, idx)
    await verify_career_structure(evaluator, uni_node, uni, idx)
    await verify_career_offerings(evaluator, uni_node, uni, idx)
    await verify_graduate_outcomes(evaluator, uni_node, uni, idx)


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
    Evaluate an answer for the comprehensive U.S. universities career preparation task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation across universities
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

    # 1) Extract structured universities information
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Normalize to exactly four universities (first four; pad with empty if fewer)
    universities: List[UniversityInfo] = list(extracted.universities or [])
    universities = universities[:4]
    while len(universities) < 4:
        universities.append(UniversityInfo())

    # 2) Build verification tree and verify each university
    for i in range(4):
        await verify_university(evaluator, root, universities[i], i)

    # 3) Return evaluation summary
    return evaluator.get_summary()