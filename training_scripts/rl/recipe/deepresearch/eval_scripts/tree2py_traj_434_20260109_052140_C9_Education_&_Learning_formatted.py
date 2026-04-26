import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "online_curriculum_instruction_masters_eval"
TASK_DESCRIPTION = (
    "Identify an online Master of Education (M.Ed.) or Master of Science (M.S.) degree program in Curriculum and Instruction "
    "from a U.S. university that meets ALL mandatory requirements and as many preferred requirements as possible. "
    "Provide program name, institution name, and URL evidence for each requirement."
)

ALLOWED_REGIONAL_ACCREDITORS = [
    "NECHE", "MSCHE", "HLC", "SACSCOC", "WSCUC", "NWCCU", "WASC", "WASC Senior College and University Commission"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    # Core identification
    program_name: Optional[str] = None
    institution_name: Optional[str] = None
    degree_type: Optional[str] = None  # e.g., "M.Ed.", "Master of Education", "M.S.", "Master of Science"
    field: Optional[str] = None  # e.g., "Curriculum and Instruction"
    program_page_url: Optional[str] = None  # main program page if available

    # Evidence URLs per requirement group
    accreditation_urls: List[str] = Field(default_factory=list)
    format_urls: List[str] = Field(default_factory=list)
    testing_urls: List[str] = Field(default_factory=list)
    credits_urls: List[str] = Field(default_factory=list)
    gpa_urls: List[str] = Field(default_factory=list)
    experience_urls: List[str] = Field(default_factory=list)

    transfer_urls: List[str] = Field(default_factory=list)
    tuition_urls: List[str] = Field(default_factory=list)
    duration_urls: List[str] = Field(default_factory=list)
    capstone_urls: List[str] = Field(default_factory=list)
    support_urls: List[str] = Field(default_factory=list)
    advising_urls: List[str] = Field(default_factory=list)
    library_urls: List[str] = Field(default_factory=list)
    start_dates_urls: List[str] = Field(default_factory=list)

    # Optional extracted values/descriptions to help phrasing claims (kept as strings for robustness)
    regional_accreditor_name: Optional[str] = None
    fully_online_desc: Optional[str] = None
    asynchronous_desc: Optional[str] = None

    total_credits: Optional[str] = None
    min_gpa: Optional[str] = None
    experience_required_description: Optional[str] = None

    transfer_credits_accepted: Optional[str] = None
    tuition_per_credit: Optional[str] = None
    part_time_duration: Optional[str] = None
    capstone_or_thesis: Optional[str] = None
    support_availability: Optional[str] = None
    advising_availability: Optional[str] = None
    library_access_desc: Optional[str] = None
    start_dates_desc: Optional[str] = None

    us_location_desc: Optional[str] = None  # e.g., state, "United States", etc.


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
    You will extract details for a single online master's program described in the answer. If multiple programs are mentioned, extract the first clearly described program focused on Curriculum and Instruction.

    Extract the following fields exactly as they appear in the answer (use strings; if missing, return null; for URLs return an array, empty if none):

    Core identification:
    - program_name: The exact program name.
    - institution_name: The university name.
    - degree_type: The degree type text (e.g., "M.Ed.", "Master of Education", "M.S.", "Master of Science").
    - field: The field area (should be "Curriculum and Instruction" or equivalent).
    - program_page_url: Main program webpage URL if provided.

    Evidence URLs for each requirement (extract all URLs explicitly provided for each category):
    - accreditation_urls: URLs showing the institution’s regional accreditation.
    - format_urls: URLs supporting that the program is fully online and primarily asynchronous.
    - testing_urls: URLs supporting that GRE/GMAT is not required.
    - credits_urls: URLs supporting total program credit hours.
    - gpa_urls: URLs supporting minimum GPA requirement.
    - experience_urls: URLs supporting any professional experience requirement (or that none is required).

    Preferred requirement evidence URLs (if available, else empty arrays):
    - transfer_urls
    - tuition_urls
    - duration_urls
    - capstone_urls
    - support_urls
    - advising_urls
    - library_urls
    - start_dates_urls

    Optional descriptive values to help verification (strings; copy from the answer when available):
    - regional_accreditor_name
    - fully_online_desc
    - asynchronous_desc
    - total_credits
    - min_gpa
    - experience_required_description
    - transfer_credits_accepted
    - tuition_per_credit
    - part_time_duration
    - capstone_or_thesis
    - support_availability
    - advising_availability
    - library_access_desc
    - start_dates_desc
    - us_location_desc

    IMPORTANT:
    - Only extract URLs explicitly present in the answer.
    - Do not fabricate values or URLs. If something is not present, return null (or an empty array for URL lists).
    - Keep numbers and thresholds as raw strings if present (e.g., "36 credits", "3.0 GPA", "12 credits", "$750/credit").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merged_sources(primary_urls: List[str], fallback_url: Optional[str]) -> List[str] | None:
    """Return a list of URLs prioritizing primary_urls; if empty, use fallback_url (if any)."""
    if primary_urls and len(primary_urls) > 0:
        return primary_urls
    if fallback_url:
        return [fallback_url]
    return None


async def add_url_evidence_presence_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    urls: List[str],
    critical: bool
):
    """Add a custom node to check presence of URL evidence."""
    evaluator.add_custom_node(
        result=(urls is not None and len(urls) > 0),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )


async def add_verified_leaf(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str] | None,
    critical: bool,
    additional_instruction: str
):
    """Create a leaf node and verify a claim using provided URLs (if any)."""
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Build verification tree                                                     #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, info: ProgramExtraction):
    # Top-level grouping under root
    suitable_program = evaluator.add_parallel(
        id="suitable_program",
        desc="Identify one online M.Ed. or M.S. program in Curriculum and Instruction from a U.S. university and provide program/institution name plus URL evidence for each requirement",
        parent=evaluator.root,
        critical=False  # Set non-critical to allow mixing critical (mandatory) and non-critical (preferred) children
    )

    # ------------------------ Program identity (CRITICAL) -----------------------
    program_identity = evaluator.add_parallel(
        id="program_identity",
        desc="Program identification matches the requested degree type, field, and U.S. institution, and includes program/institution names",
        parent=suitable_program,
        critical=True
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(info.program_name and info.program_name.strip()),
        id="program_name_provided",
        desc="Program name is provided",
        parent=program_identity,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.institution_name and info.institution_name.strip()),
        id="institution_name_provided",
        desc="Institution name is provided",
        parent=program_identity,
        critical=True
    )

    # Institution is a U.S. university
    inst_sources = merged_sources([], info.program_page_url)
    await add_verified_leaf(
        evaluator,
        program_identity,
        "institution_is_us_university",
        "Institution is a U.S. university",
        claim=f"The institution '{info.institution_name or 'the institution'}' is a U.S. university (located in the United States).",
        urls=inst_sources,
        critical=True,
        additional_instruction="Look for the institution's location on the page (city/state or 'United States'). If clearly in the U.S., pass."
    )

    # Degree type correct (M.Ed. or M.S.)
    await add_verified_leaf(
        evaluator,
        program_identity,
        "degree_type_correct",
        "Program is a Master of Education (M.Ed.) or Master of Science (M.S.)",
        claim=f"The program is a Master's degree of type {info.degree_type or 'M.Ed. or M.S.'}, and it is either a Master of Education (M.Ed.) or a Master of Science (M.S.).",
        urls=merged_sources([], info.program_page_url),
        critical=True,
        additional_instruction="Accept reasonable variants like 'Master of Education', 'M.Ed.', 'Master of Science', 'M.S.'."
    )

    # Field correct (Curriculum and Instruction)
    await add_verified_leaf(
        evaluator,
        program_identity,
        "field_correct",
        "Program is in Curriculum and Instruction",
        claim=f"The program's field is Curriculum and Instruction (or an equivalently named 'Curriculum & Instruction').",
        urls=merged_sources([], info.program_page_url),
        critical=True,
        additional_instruction="Allow small naming variations like 'Curriculum & Instruction'. The program should be clearly in this field."
    )

    # ------------------------ Regional accreditation (CRITICAL) -----------------
    regional_accreditation = evaluator.add_parallel(
        id="regional_accreditation",
        desc="Institution holds regional accreditation from an allowed regional accreditor, with URL evidence",
        parent=suitable_program,
        critical=True
    )

    # Evidence presence
    await add_url_evidence_presence_node(
        evaluator,
        regional_accreditation,
        "accreditation_url_evidence",
        "Provides URL evidence supporting the institution's regional accreditation",
        urls=info.accreditation_urls,
        critical=True
    )

    # Accreditation type check
    accred_name = info.regional_accreditor_name or "a recognized U.S. regional accreditor"
    await add_verified_leaf(
        evaluator,
        regional_accreditation,
        "accreditation_type",
        "Institution is accredited by one of the specified regional accreditors (NECHE, MSCHE, HLC, SACSCOC, WSCUC, NWCCU, or WASC)",
        claim=f"The institution is accredited by {accred_name}, which is one of the recognized U.S. regional accreditors (NECHE, MSCHE, HLC, SACSCOC, WSCUC, NWCCU, or WASC).",
        urls=merged_sources(info.accreditation_urls, info.program_page_url),
        critical=True,
        additional_instruction="Confirm the accreditor is among: NECHE, MSCHE, HLC, SACSCOC, WSCUC, NWCCU (WASC Senior College is acceptable)."
    )

    # ------------------------ Program format (CRITICAL) -------------------------
    program_format = evaluator.add_parallel(
        id="program_format",
        desc="Program is fully online and primarily asynchronous, with URL evidence",
        parent=suitable_program,
        critical=True
    )

    await add_url_evidence_presence_node(
        evaluator,
        program_format,
        "format_url_evidence",
        "Provides URL evidence supporting the online and asynchronous format claims",
        urls=info.format_urls,
        critical=True
    )

    await add_verified_leaf(
        evaluator,
        program_format,
        "fully_online_no_residency",
        "Program is 100% online with no required on-campus attendance/residency",
        claim="The program is 100% online with no required on-campus attendance or residency.",
        urls=merged_sources(info.format_urls, info.program_page_url),
        critical=True,
        additional_instruction="Pass only if the page clearly indicates fully online with no required campus visits or residencies."
    )

    await add_verified_leaf(
        evaluator,
        program_format,
        "primarily_asynchronous",
        "Courses are primarily asynchronous (no fixed synchronous meeting times required as the primary mode)",
        claim="The program's courses are primarily asynchronous, allowing students to complete work on their own schedule without fixed weekly live meeting times as the primary mode.",
        urls=merged_sources(info.format_urls, info.program_page_url),
        critical=True,
        additional_instruction="If they allow some optional synchronous sessions but primarily run asynchronously, this passes."
    )

    # ------------------------ Admission testing (CRITICAL) ----------------------
    admission_testing = evaluator.add_parallel(
        id="admission_testing",
        desc="No GRE/GMAT required, with URL evidence",
        parent=suitable_program,
        critical=True
    )

    await add_url_evidence_presence_node(
        evaluator,
        admission_testing,
        "testing_url_evidence",
        "Provides URL evidence supporting that GRE/GMAT is not required",
        urls=info.testing_urls,
        critical=True
    )

    await add_verified_leaf(
        evaluator,
        admission_testing,
        "no_gre_gmat",
        "GRE or GMAT is not required for admission",
        claim="GRE or GMAT scores are not required for admission to this program.",
        urls=merged_sources(info.testing_urls, info.program_page_url),
        critical=True,
        additional_instruction="If the page says 'GRE not required' or 'waived for all applicants', pass. If optional, still pass as 'not required'."
    )

    # ------------------------ Credit hours (CRITICAL) ---------------------------
    credit_hours = evaluator.add_parallel(
        id="credit_hours",
        desc="Total credits fall within 30–48, with URL evidence",
        parent=suitable_program,
        critical=True
    )

    await add_url_evidence_presence_node(
        evaluator,
        credit_hours,
        "credits_url_evidence",
        "Provides URL evidence supporting the total credit requirement",
        urls=info.credits_urls,
        critical=True
    )

    await add_verified_leaf(
        evaluator,
        credit_hours,
        "total_credits_range",
        "Total program credit hours are between 30 and 48 (inclusive)",
        claim="The program requires a total between 30 and 48 credit hours (inclusive).",
        urls=merged_sources(info.credits_urls, info.program_page_url),
        critical=True,
        additional_instruction="Check the total program credits. If any track or standard path is within 30–48, accept."
    )

    # ------------------------ Admission GPA (CRITICAL) --------------------------
    admission_gpa = evaluator.add_parallel(
        id="admission_gpa",
        desc="Minimum GPA requirement is 3.0 or lower, with URL evidence",
        parent=suitable_program,
        critical=True
    )

    await add_url_evidence_presence_node(
        evaluator,
        admission_gpa,
        "gpa_url_evidence",
        "Provides URL evidence supporting the minimum GPA requirement",
        urls=info.gpa_urls,
        critical=True
    )

    await add_verified_leaf(
        evaluator,
        admission_gpa,
        "min_gpa_at_most_3_0",
        "Minimum GPA requirement for admission is 3.0 or lower (4.0 scale)",
        claim="The minimum GPA requirement for admission is at most 3.0 on a 4.0 scale.",
        urls=merged_sources(info.gpa_urls, info.program_page_url),
        critical=True,
        additional_instruction="If the page states 3.0, 2.75, or similar thresholds (<= 3.0) for admission, pass. If only higher thresholds (e.g., 3.25), fail."
    )

    # ------------------------ Professional experience (CRITICAL) ----------------
    professional_experience = evaluator.add_parallel(
        id="professional_experience",
        desc="If experience is required, it is 1 year or less, with URL evidence",
        parent=suitable_program,
        critical=True
    )

    await add_url_evidence_presence_node(
        evaluator,
        professional_experience,
        "experience_url_evidence",
        "Provides URL evidence supporting the experience requirement (or lack thereof)",
        urls=info.experience_urls,
        critical=True
    )

    await add_verified_leaf(
        evaluator,
        professional_experience,
        "experience_at_most_1_year_if_required",
        "If professional teaching/educational experience is required, the requirement is 1 year or less",
        claim="If any professional teaching or educational experience is required for admission, the requirement is 1 year or less (or no experience required).",
        urls=merged_sources(info.experience_urls, info.program_page_url),
        critical=True,
        additional_instruction="If the page indicates no experience required, this passes. If experience required is 1 year or less, pass; if 2+ years, fail."
    )

    # ============================ PREFERRED (non-critical) ======================

    # Transfer credit policy
    transfer_credit_policy = evaluator.add_parallel(
        id="transfer_credit_policy",
        desc="Preferred: accepts sufficient transfer credits, with URL evidence",
        parent=suitable_program,
        critical=False
    )

    await add_url_evidence_presence_node(
        evaluator,
        transfer_credit_policy,
        "transfer_url_evidence",
        "Provides URL evidence supporting transfer credit acceptance policy",
        urls=info.transfer_urls,
        critical=False
    )

    await add_verified_leaf(
        evaluator,
        transfer_credit_policy,
        "transfer_credit_threshold",
        "Accepts at least 12 transfer credits OR at least 25% of total program credits",
        claim="The program accepts at least 12 transfer credits or at least 25% of the total required credits.",
        urls=merged_sources(info.transfer_urls, info.program_page_url),
        critical=False,
        additional_instruction="Look for statements like 'up to 12 credits' or 'up to 25% of the program' transferable. Either threshold suffices."
    )

    # Tuition cost
    tuition_cost = evaluator.add_parallel(
        id="tuition_cost",
        desc="Preferred: tuition per credit hour is $800 or less, with URL evidence",
        parent=suitable_program,
        critical=False
    )

    await add_url_evidence_presence_node(
        evaluator,
        tuition_cost,
        "tuition_url_evidence",
        "Provides URL evidence supporting tuition per credit hour",
        urls=info.tuition_urls,
        critical=False
    )

    await add_verified_leaf(
        evaluator,
        tuition_cost,
        "tuition_per_credit_at_most_800",
        "Tuition cost is $800 or less per credit hour",
        claim="Tuition is $800 per credit hour or less for this online program.",
        urls=merged_sources(info.tuition_urls, info.program_page_url),
        critical=False,
        additional_instruction="If multiple rates are listed (e.g., in-state/out-of-state/online), consider the relevant online rate. Pass if any legitimate per-credit rate is <= $800."
    )

    # Program duration
    program_duration = evaluator.add_parallel(
        id="program_duration",
        desc="Preferred: completion within 24 months part-time, with URL evidence",
        parent=suitable_program,
        critical=False
    )

    await add_url_evidence_presence_node(
        evaluator,
        program_duration,
        "duration_url_evidence",
        "Provides URL evidence supporting the completion timeline",
        urls=info.duration_urls,
        critical=False
    )

    await add_verified_leaf(
        evaluator,
        program_duration,
        "part_time_within_24_months",
        "Program can be completed within 24 months part-time",
        claim="This program can be completed within 24 months (2 years) on a part-time schedule.",
        urls=merged_sources(info.duration_urls, info.program_page_url),
        critical=False,
        additional_instruction="Check stated time-to-completion estimates. If a typical part-time pace finishes in <= 24 months, pass."
    )

    # Capstone requirement
    capstone_requirement = evaluator.add_parallel(
        id="capstone_requirement",
        desc="Preferred: has capstone/thesis culminating experience, with URL evidence",
        parent=suitable_program,
        critical=False
    )

    await add_url_evidence_presence_node(
        evaluator,
        capstone_requirement,
        "capstone_url_evidence",
        "Provides URL evidence supporting the culminating experience requirement",
        urls=info.capstone_urls,
        critical=False
    )

    await add_verified_leaf(
        evaluator,
        capstone_requirement,
        "capstone_or_thesis_present",
        "Requires a capstone project, thesis, or equivalent culminating experience",
        claim="The program requires a culminating experience such as a capstone project or thesis.",
        urls=merged_sources(info.capstone_urls, info.program_page_url),
        critical=False,
        additional_instruction="Accept capstone, thesis, portfolio, comprehensive project, or similar culminating requirements."
    )

    # Student support availability
    student_support = evaluator.add_parallel(
        id="student_support",
        desc="Preferred: 24/7 or extended-hours online support, with URL evidence",
        parent=suitable_program,
        critical=False
    )

    await add_url_evidence_presence_node(
        evaluator,
        student_support,
        "support_url_evidence",
        "Provides URL evidence supporting support service availability",
        urls=info.support_urls,
        critical=False
    )

    await add_verified_leaf(
        evaluator,
        student_support,
        "support_24_7_or_extended_hours",
        "Provides 24/7 or extended-hours online student support services",
        claim="Online student support services are available 24/7 or during extended hours (e.g., evenings/weekends).",
        urls=merged_sources(info.support_urls, info.program_page_url),
        critical=False,
        additional_instruction="Look for 24/7 helpdesk, tutoring, technical support, library chat, etc., or clear extended-hours coverage."
    )

    # Academic advising or mentors
    academic_advising = evaluator.add_parallel(
        id="academic_advising",
        desc="Preferred: advisors or mentors available, with URL evidence",
        parent=suitable_program,
        critical=False
    )

    await add_url_evidence_presence_node(
        evaluator,
        academic_advising,
        "advising_url_evidence",
        "Provides URL evidence supporting advising/mentoring access",
        urls=info.advising_urls,
        critical=False
    )

    await add_verified_leaf(
        evaluator,
        academic_advising,
        "advisors_or_mentors_available",
        "Provides access to academic advisors or program mentors",
        claim="Students have access to academic advisors or program mentors.",
        urls=merged_sources(info.advising_urls, info.program_page_url),
        critical=False,
        additional_instruction="Any clearly described advising or mentoring support for the program or online students qualifies."
    )

    # Library access
    library_access = evaluator.add_parallel(
        id="library_access",
        desc="Preferred: online library access, with URL evidence",
        parent=suitable_program,
        critical=False
    )

    await add_url_evidence_presence_node(
        evaluator,
        library_access,
        "library_url_evidence",
        "Provides URL evidence supporting online library access",
        urls=info.library_urls,
        critical=False
    )

    await add_verified_leaf(
        evaluator,
        library_access,
        "online_library_access",
        "Provides access to library resources and databases to online students",
        claim="Online students receive access to university library resources and databases.",
        urls=merged_sources(info.library_urls, info.program_page_url),
        critical=False,
        additional_instruction="Accept explicit statements about online/remote access to library databases, e-resources, and librarian support."
    )

    # Start dates
    start_dates = evaluator.add_parallel(
        id="start_dates",
        desc="Preferred: at least 4 start dates per year, with URL evidence",
        parent=suitable_program,
        critical=False
    )

    await add_url_evidence_presence_node(
        evaluator,
        start_dates,
        "start_dates_url_evidence",
        "Provides URL evidence supporting start-date frequency",
        urls=info.start_dates_urls,
        critical=False
    )

    await add_verified_leaf(
        evaluator,
        start_dates,
        "at_least_4_start_dates_per_year",
        "Offers at least 4 start dates per year (quarterly or more frequent)",
        claim="The program offers at least four start dates per year (quarterly or more frequent).",
        urls=merged_sources(info.start_dates_urls, info.program_page_url),
        critical=False,
        additional_instruction="Quarterly, monthly, or rolling starts that effectively allow 4+ start points per year should pass."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Entry point to evaluate an agent's answer for the Curriculum & Instruction online master's program task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction"
    )

    # Record allowed accreditors for transparency
    evaluator.add_custom_info(
        info={"allowed_regional_accreditors": ALLOWED_REGIONAL_ACCREDITORS},
        info_type="config",
        info_name="accreditation_policy"
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()