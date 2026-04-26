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
TASK_ID = "edtech_2024_jan2025_degrees"
TASK_DESCRIPTION = (
    "Identify a U.S. university that announced a partnership in 2024 with an education technology company to launch "
    "at least 15 new online degree programs (including both undergraduate and graduate levels) specifically designed "
    "for working adult learners, where these programs are scheduled to begin classes in January 2025. For this university, "
    "provide: (1) the name of the education technology partner company, (2) the exact number of new programs being launched, "
    "(3) the university's regional accrediting body, (4) the specific date when the first classes begin, and (5) the application deadline for these programs."
)


# --------------------------------------------------------------------------- #
# Known regional accreditors (normalized)                                     #
# --------------------------------------------------------------------------- #
KNOWN_REGIONAL_ACCREDITORS = [
    # Full names
    "higher learning commission",
    "middle states commission on higher education",
    "new england commission of higher education",
    "southern association of colleges and schools commission on colleges",
    "wasc senior college and university commission",
    "northwest commission on colleges and universities",
    # Common abbreviations
    "hlc",
    "msche",
    "neche",
    "sacs",        # sometimes appears without full 'sacscoc'
    "sacscoc",
    "wscuc",
    "nwccu",
]


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s\-\._,;:\/\\]+", " ", s)
    return s


def is_recognized_regional_accreditor(name: Optional[str]) -> bool:
    if not name:
        return False
    norm = _normalize_text(name)
    return any(token in norm for token in KNOWN_REGIONAL_ACCREDITORS)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PartnershipInfo(BaseModel):
    # Core entities
    university: Optional[str] = None
    partner_company: Optional[str] = None

    # Numbers and constraints
    program_count: Optional[str] = None  # keep as string to maximize compatibility

    # Program characteristics
    degree_levels: List[str] = Field(default_factory=list)  # e.g., ["undergraduate", "graduate"]
    online_degree_programs: Optional[str] = None           # freeform phrase as stated in answer
    targets_working_adults: Optional[str] = None           # phrase indicating target audience
    workforce_focus: Optional[str] = None                  # phrase indicating workforce focus

    # Timing
    announcement_year: Optional[str] = None                # e.g., "2024"
    class_start_date: Optional[str] = None                 # exact date string
    application_deadline: Optional[str] = None             # exact date string

    # Accreditation
    regional_accreditor: Optional[str] = None

    # Sources cited in the answer
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_partnership_info() -> str:
    return """
    Extract structured information about a single university partnership described in the answer.

    Required fields (return null for any missing field):
    - university: The name of the U.S. university identified.
    - partner_company: The name of the education technology partner company.
    - program_count: The exact number of new programs being launched (extract exactly as stated in the answer; keep it as text, e.g., "15", "over 20", "twenty", etc.).
    - degree_levels: A list of degree levels explicitly mentioned for the new programs; include any of ["undergraduate", "graduate", "associate", "bachelor's", "master's", "doctoral", "phd"]; return [] if not stated.
    - online_degree_programs: The exact phrase or sentence indicating that these are online degree programs (not certificates); return null if not stated.
    - targets_working_adults: The phrase indicating the programs are designed for working adult learners or nontraditional students; return null if not stated.
    - workforce_focus: The phrase indicating the programs are workforce-relevant or workforce-focused; return null if not stated.
    - announcement_year: The year the partnership was announced or established (e.g., "2024"); return null if not stated.
    - class_start_date: The exact date when the first classes begin (e.g., "January 13, 2025", "2025-01-13", "1/13/2025"); return null if not stated.
    - application_deadline: The exact application deadline date for these programs (e.g., "December 1, 2024"); return null if not stated.
    - regional_accreditor: The university's regional accrediting body (e.g., "Higher Learning Commission"); return null if not stated.
    - sources: An array with ALL URLs explicitly provided in the answer text that relate to this partnership and its details (press releases, news, partner/company pages, accreditation pages, program pages, FAQs, etc.). Extract the actual URLs; include Google Docs or PDFs if present. If the answer mentions a source but no actual URL, do not invent one.

    Notes:
    - Extract EXACT text as it appears in the answer for each field.
    - Do not infer or add information that is not explicitly in the answer, except for URLs which you must extract as they are provided.
    - For degree_levels, normalize to lowercase single tokens when possible (e.g., 'undergraduate', 'graduate'); include multiple entries if both are mentioned.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, parent_node, info: PartnershipInfo) -> None:
    """
    Construct the verification tree according to the rubric and run checks.
    """
    # Top-level critical node aggregating all subcriteria
    main_node = evaluator.add_parallel(
        id="qualifying_university_partnership_and_requested_details",
        desc="Answer identifies one U.S. university and verifies it meets the 2024 edtech-partnership and January 2025 online-program launch constraints, while providing the five requested details (partner, program count, accreditor, class start date, application deadline).",
        parent=parent_node,
        critical=True
    )

    # Shared sources
    srcs: List[str] = info.sources if info.sources else []

    # 1) University Identification (critical)
    uni_node = evaluator.add_parallel(
        id="university_identification",
        desc="The response identifies the university and confirms it is a U.S. higher education institution.",
        parent=main_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.university and info.university.strip()),
        id="university_provided",
        desc="A university name is provided in the response.",
        parent=uni_node,
        critical=True
    )

    uni_us_leaf = evaluator.add_leaf(
        id="university_is_us_institution",
        desc="The identified university is a U.S. higher education institution.",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The institution '{info.university}' is a U.S. higher education university/institution.",
        node=uni_us_leaf,
        sources=srcs,
        additional_instruction="Allow verification from the university's official pages, Wikipedia, or press releases. Accept clear indications such as location in the United States or statements of U.S. accreditation."
    )

    # 2) Edtech Partner Provided and Is Edtech (critical)
    partner_node = evaluator.add_parallel(
        id="edtech_partner_provided_and_is_edtech",
        desc="The response provides the name of the education-technology partner company and the partner is described as an education technology company.",
        parent=main_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.partner_company and info.partner_company.strip()),
        id="partner_provided",
        desc="An education-technology partner company name is provided.",
        parent=partner_node,
        critical=True
    )

    partner_edtech_leaf = evaluator.add_leaf(
        id="partner_is_edtech",
        desc="The named partner is an education technology company.",
        parent=partner_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The company '{info.partner_company}' is an education technology (edtech) company or platform provider (including OPM-type edtech).",
        node=partner_edtech_leaf,
        sources=srcs,
        additional_instruction="Accept descriptions such as 'edtech', 'education technology company', 'online learning platform', 'digital education company', or 'online program manager (OPM)' as satisfying this criterion."
    )

    # 3) Partnership Announced in 2024 (critical)
    announced_2024_leaf = evaluator.add_leaf(
        id="partnership_announced_in_2024",
        desc="The response states the partnership was announced or established in 2024.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The partnership between '{info.university}' and '{info.partner_company}' was announced or established in 2024.",
        node=announced_2024_leaf,
        sources=srcs,
        additional_instruction="Look for press releases or news mentioning '2024', or language like 'announced in 2024', 'in 2024', 'established in 2024', or specific 2024 dates."
    )

    # 4) Programs Are Online Degrees (critical)
    online_degrees_leaf = evaluator.add_leaf(
        id="programs_are_online_degrees",
        desc="The response indicates the launched programs are online degree programs (not only certificates or non-degree offerings).",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The partnership involves launching online degree programs (not merely certificates or non-degree offerings) at '{info.university}'.",
        node=online_degrees_leaf,
        sources=srcs,
        additional_instruction="Confirm phrasing like 'online degree programs', 'online bachelor's/master's degrees', or explicit statements that they are degree-granting programs."
    )

    # 5) Program Count Exact and At Least 15 (critical)
    count_node = evaluator.add_parallel(
        id="program_count_exact_and_at_least_15",
        desc="The response provides the exact number of new programs being launched and that number is at least 15.",
        parent=main_node,
        critical=True
    )

    count_exact_leaf = evaluator.add_leaf(
        id="program_count_exact_supported",
        desc="The provided number of new programs is supported by the cited sources.",
        parent=count_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The number of new online degree programs being launched is exactly '{info.program_count}'.",
        node=count_exact_leaf,
        sources=srcs,
        additional_instruction="Verify the exact count (e.g., '15 programs', '20 new degrees'). Allow minor textual variants and numeric formatting but the count should be explicitly supported."
    )

    count_at_least_15_leaf = evaluator.add_leaf(
        id="program_count_at_least_15",
        desc="The number of new programs is at least 15.",
        parent=count_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program count '{info.program_count}' indicates at least 15 programs.",
        node=count_at_least_15_leaf,
        additional_instruction="Interpret the text/number naturally (e.g., 'fifteen', '15', 'at least 18', '20') and judge whether it clearly means 15 or more."
    )

    # 6) Includes Undergraduate and Graduate Levels (critical)
    levels_leaf = evaluator.add_leaf(
        id="includes_undergrad_and_graduate_levels",
        desc="The response confirms the program set includes both undergraduate and graduate degree levels.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The announced online degree programs at '{info.university}' include both undergraduate and graduate levels.",
        node=levels_leaf,
        sources=srcs,
        additional_instruction="Look for explicit mentions of 'undergraduate and graduate', or both bachelor's and master's/graduate degrees. Allow equivalents (e.g., bachelor's = undergraduate; master's/doctoral = graduate)."
    )

    # 7) Targets Working Adult Learners (critical)
    adult_target_leaf = evaluator.add_leaf(
        id="targets_working_adult_learners",
        desc="The response explicitly indicates the programs are designed for working adult learners/nontraditional students.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The programs are designed for working adult learners or nontraditional students.",
        node=adult_target_leaf,
        sources=srcs,
        additional_instruction="Accept wording such as 'working adults', 'adult learners', 'nontraditional learners', 'busy professionals', or similar."
    )

    # 8) Workforce-Focused Orientation (critical)
    workforce_leaf = evaluator.add_leaf(
        id="workforce_focused_orientation",
        desc="The response indicates the programs are described as workforce-relevant or workforce-focused.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The programs are described as workforce-relevant or workforce-focused.",
        node=workforce_leaf,
        sources=srcs,
        additional_instruction="Look for terms such as 'workforce-aligned', 'career-relevant', 'career-focused', 'industry-aligned', 'skills-focused', or similar phrasing."
    )

    # 9) Regional Accreditor Provided and Regional (critical)
    accred_node = evaluator.add_parallel(
        id="regional_accreditor_provided_and_regional",
        desc="The response provides the university’s regional accrediting body and it is a recognized U.S. regional accreditor.",
        parent=main_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.regional_accreditor and info.regional_accreditor.strip()),
        id="accreditor_provided",
        desc="A regional accrediting body name is provided.",
        parent=accred_node,
        critical=True
    )

    accred_supported_leaf = evaluator.add_leaf(
        id="accreditor_supported_by_sources",
        desc="The cited sources support that the university is accredited by the named accrediting body.",
        parent=accred_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{info.university}' is accredited by '{info.regional_accreditor}'.",
        node=accred_supported_leaf,
        sources=srcs,
        additional_instruction="Accept university accreditation pages, accreditor directories, or authoritative sources that explicitly state the accreditor for the university."
    )

    evaluator.add_custom_node(
        result=is_recognized_regional_accreditor(info.regional_accreditor),
        id="accreditor_is_recognized_regional",
        desc="The named accrediting body is a recognized U.S. regional accreditor.",
        parent=accred_node,
        critical=True
    )

    # 10) Classes Start Date Exact and In January 2025 (critical)
    start_node = evaluator.add_parallel(
        id="classes_start_date_exact_and_in_january_2025",
        desc="The response provides the specific first day of classes as an exact date, and it falls in January 2025.",
        parent=main_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.class_start_date and info.class_start_date.strip()),
        id="start_date_provided",
        desc="A specific first day of classes is provided.",
        parent=start_node,
        critical=True
    )

    start_supported_leaf = evaluator.add_leaf(
        id="start_date_supported_by_sources",
        desc="The specific first day of classes is supported by the cited sources.",
        parent=start_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first day of classes for these programs is {info.class_start_date}.",
        node=start_supported_leaf,
        sources=srcs,
        additional_instruction="The source should indicate the first day that classes begin for this launch/cohort."
    )

    start_in_jan_2025_leaf = evaluator.add_leaf(
        id="start_date_in_january_2025",
        desc="The start date falls in January 2025 and is an exact date (day included).",
        parent=start_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The date '{info.class_start_date}' is an exact calendar date in January 2025 (i.e., includes a day in January 2025).",
        node=start_in_jan_2025_leaf,
        additional_instruction="Check that the string clearly includes a day-of-month and the month is January and year is 2025."
    )

    # 11) Application Deadline Provided and Before Start (critical)
    deadline_node = evaluator.add_parallel(
        id="application_deadline_provided_and_before_start",
        desc="The response provides the application deadline date for these programs, and it is before the first class start date.",
        parent=main_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.application_deadline and info.application_deadline.strip()),
        id="deadline_provided",
        desc="An application deadline date is provided.",
        parent=deadline_node,
        critical=True
    )

    deadline_supported_leaf = evaluator.add_leaf(
        id="deadline_supported_by_sources",
        desc="The application deadline is supported by the cited sources.",
        parent=deadline_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The application deadline for these programs is {info.application_deadline}.",
        node=deadline_supported_leaf,
        sources=srcs,
        additional_instruction="The cited page should clearly list an application deadline for this cohort/start."
    )

    deadline_before_start_leaf = evaluator.add_leaf(
        id="deadline_before_start",
        desc="The application deadline occurs before the first class start date.",
        parent=deadline_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The application deadline '{info.application_deadline}' is before the class start date '{info.class_start_date}'.",
        node=deadline_before_start_leaf,
        additional_instruction="Interpret and compare the two dates logically; allow standard date formats and canonicalize if needed."
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
    Evaluate an answer against the rubric for the edtech-2024/Jan-2025 online degree program launch task.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_partnership_info(),
        template_class=PartnershipInfo,
        extraction_name="partnership_info",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()