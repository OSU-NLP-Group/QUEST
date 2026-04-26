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
TASK_ID = "academic_planning_2026_2027"
TASK_DESCRIPTION = """You are an early-career researcher in the field of education planning your academic trajectory for 2026-2027. You aim to build a strong academic profile before applying to PhD programs for Fall 2027 admission.

Your goals include:
1. Presenting at two major education research conferences in 2026 (one in spring, one in fall)
2. Submitting a manuscript to a peer-reviewed journal with sufficient time for the peer review process to be completed (or at least reach 'revise and resubmit' status) before PhD application deadlines
3. Applying to a PhD program in education or a related field at a research university for Fall 2027 admission
4. Identifying a relevant postdoctoral fellowship opportunity for future career planning

For each of these goals, identify specific opportunities and provide:

For the two conferences:
- The conference name, location, and exact dates in 2026
- The submission deadline (or reasonable estimate based on conference timing)
- Verification URL

For the journal submission:
- The journal name and a quality indicator (if available)
- A proposed submission timeline that accounts for:
  - Peer review duration (typically 12-14 weeks for initial decision)
  - Potential revision time (3-6 weeks)
  - PhD application deadlines in December 2026
- Verification URL

For the PhD program:
- University name and program name
- Minimum GPA requirement
- Number of required recommendation letters
- Application deadline for Fall 2027 admission
- Timeline for requesting recommendation letters (typically 2-3 months before deadline)
- Verification URLs

For the fellowship:
- Fellowship program name
- Eligibility requirements
- Application deadline
- Funding information (if available)
- Verification URL

Your answer should demonstrate understanding of how these various deadlines and requirements must be coordinated to create a feasible academic timeline for 2026-2027.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConferenceItem(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    submission_deadline: Optional[str] = None
    submission_deadline_is_estimate: Optional[bool] = None
    verification_url: Optional[str] = None
    submission_deadline_url: Optional[str] = None


class ConferencesExtraction(BaseModel):
    spring: Optional[ConferenceItem] = None
    fall: Optional[ConferenceItem] = None


class JournalPlanExtraction(BaseModel):
    journal_name: Optional[str] = None
    quality_indicator: Optional[str] = None  # e.g., impact factor, quartile, indexing; or "not available"
    verification_url: Optional[str] = None
    proposed_submission_timing: Optional[str] = None  # a date or date-range (e.g., "May 2026", "late April 2026")
    timeline_text: Optional[str] = None  # free text summary if present
    mentions_initial_decision_12_14_weeks: Optional[bool] = None
    mentions_revision_3_6_weeks: Optional[bool] = None
    targets_before_early_dec_2026: Optional[bool] = None  # True if plan aims decision/R&R before Dec 1–15, 2026


class PhDProgramExtraction(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    field_relevance_note: Optional[str] = None  # text supporting that it's in education or related field
    minimum_gpa: Optional[str] = None
    letters_required: Optional[str] = None  # keep string to be robust (e.g., "3")
    application_deadline_fall_2027: Optional[str] = None
    recommendation_request_timeline: Optional[str] = None  # e.g., "September–October 2026"
    program_urls: List[str] = Field(default_factory=list)  # program identity/info URLs
    admissions_urls: List[str] = Field(default_factory=list)  # admissions requirement/deadline URLs
    research_university_indicator: Optional[str] = None  # e.g., "Carnegie R1"
    research_indicator_url: Optional[str] = None


class FellowshipExtraction(BaseModel):
    fellowship_name: Optional[str] = None
    url: Optional[str] = None
    is_postdoctoral: Optional[bool] = None
    relevance_note: Optional[str] = None  # how it relates to education
    eligibility: Optional[str] = None
    application_deadline: Optional[str] = None
    funding_info: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
Extract two conference items for 2026 from the answer: one Spring conference and one Fall conference.

For each, extract these fields exactly as presented in the answer:
- name: The full conference name.
- location: City and state/country as given (e.g., "Philadelphia, PA" or "London, UK").
- start_date: The start date text exactly as written (e.g., "April 12, 2026" or "Apr 12, 2026").
- end_date: The end date text exactly as written.
- submission_deadline: The submission deadline text exactly as written. If the answer labels it as an estimate (e.g., "estimated deadline"), still extract the date and set the flag below.
- submission_deadline_is_estimate: true if the answer explicitly labels the submission deadline as an estimate; otherwise false or null.
- verification_url: A URL cited in the answer that verifies the conference dates/location (official site or host organization).
- submission_deadline_url: If the answer includes a distinct URL specifically for submissions/CFP deadlines, extract it; otherwise null.

Return JSON with keys:
- spring: ConferenceItem for the Spring conference (months Mar–May).
- fall: ConferenceItem for the Fall conference (months Sep–Nov).

If any field is missing in the answer, set it to null. Extract only URLs that explicitly appear in the answer.
"""


def prompt_extract_journal_plan() -> str:
    return """
Extract the journal submission plan details from the answer.

Fields:
- journal_name: The journal's name exactly as written.
- quality_indicator: A quality indicator string if provided (e.g., impact factor, Scopus/ESCI indexing, quartile); if the answer explicitly states that a quality indicator is not available/found, set this field to "not available".
- verification_url: A URL cited that verifies the journal (official site or reputable journal profile).
- proposed_submission_timing: The proposed submission timing (date or date range) exactly as written (e.g., "May 2026", "late April 2026").
- timeline_text: Any summary text the answer provides about the review/revision schedule.
- mentions_initial_decision_12_14_weeks: true if the answer explicitly mentions 12–14 weeks (or equivalent) for initial decision; else false/null.
- mentions_revision_3_6_weeks: true if the answer explicitly mentions 3–6 weeks (or equivalent) for revisions; else false/null.
- targets_before_early_dec_2026: true if the plan explicitly targets completion or at least an R&R before early December 2026 (Dec 1–15 deadlines); else false/null.

If a field is not present in the answer, set it to null. Only extract URLs that explicitly appear in the answer.
"""


def prompt_extract_phd_program() -> str:
    return """
Extract details for ONE PhD program (Fall 2027 admission) from the answer.

Fields:
- university_name: University name.
- program_name: Program or department name.
- field_relevance_note: Text from the answer indicating that the program is in education or a closely related field (e.g., educational psychology, learning sciences).
- minimum_gpa: Minimum GPA requirement exactly as written (e.g., "3.0", "3.5 on a 4.0 scale"). If not specified, null.
- letters_required: Number of required recommendation letters exactly as written (e.g., "3"). If not specified, null.
- application_deadline_fall_2027: The application deadline for Fall 2027 admission exactly as written (e.g., "December 1, 2026").
- recommendation_request_timeline: The suggested timeline for requesting recommendation letters (e.g., "2–3 months before the deadline", or a concrete month range like "September–October 2026") exactly as written.
- program_urls: A list of URL(s) cited in the answer that verify the program identity/description.
- admissions_urls: A list of URL(s) cited that verify admissions requirements/deadlines (can be the same as program URLs if the page contains both).
- research_university_indicator: If provided, a phrase like "Carnegie R1" or "research university" indicator exactly as written.
- research_indicator_url: A URL cited that verifies the research-university indicator.

Only extract URLs explicitly present in the answer. If any field is missing, set it to null (or an empty list for URL lists).
"""


def prompt_extract_fellowship() -> str:
    return """
Extract details for ONE postdoctoral fellowship from the answer.

Fields:
- fellowship_name: The fellowship program name.
- url: The verification URL for the fellowship (official program page preferred).
- is_postdoctoral: true if the answer states it's a postdoctoral fellowship; otherwise false/null.
- relevance_note: Text from the answer indicating how the fellowship supports education or education-related research/training (or a compatible broader area).
- eligibility: Eligibility requirements text exactly as written (include key points like PhD status, citizenship, years since PhD, etc.).
- application_deadline: The application deadline exactly as written.
- funding_info: Funding information if the answer provides it (e.g., stipend amount/duration); otherwise null.

Only extract URLs explicitly present in the answer. If any field is missing, set it to null.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_conference_item(
    evaluator: Evaluator,
    parent_node,
    conf: ConferenceItem,
    season: str
) -> None:
    """
    Verify a single conference item (spring or fall) under its parent node.
    season: "spring" or "fall" to check month windows.
    """
    item_id_prefix = "Spring" if season.lower() == "spring" else "Fall"
    months_desc = "March–May" if season.lower() == "spring" else "September–November"

    # Name provided (critical)
    evaluator.add_custom_node(
        result=bool(conf and conf.name and conf.name.strip()),
        id=f"{item_id_prefix}_Conference_Name",
        desc="Conference name is provided.",
        parent=parent_node,
        critical=True
    )

    # Location provided with city and state/country (critical)
    loc_leaf = evaluator.add_leaf(
        id=f"{item_id_prefix}_Conference_Location",
        desc="Conference location (city and state/country) is provided.",
        parent=parent_node,
        critical=True
    )
    loc_val = conf.location or ""
    await evaluator.verify(
        claim=f"The provided location string '{loc_val}' includes both a city and a state/country.",
        node=loc_leaf,
        additional_instruction="Judge correct if the location clearly includes both a city and a state/country (e.g., contains a comma-separated pair like 'City, ST' or 'City, Country'). If location is missing or only a single place word, mark incorrect."
    )

    # Dates in correct season and year 2026 (critical)
    dates_leaf = evaluator.add_leaf(
        id=f"{item_id_prefix}_Conference_Dates_2026",
        desc=f"Exact conference start/end dates in 2026 are provided and fall in { 'spring' if season=='spring' else 'fall' } months.",
        parent=parent_node,
        critical=True
    )
    sd = conf.start_date or ""
    ed = conf.end_date or ""
    await evaluator.verify(
        claim=f"The conference dates '{sd}' to '{ed}' occur in 2026 and fall within {months_desc}.",
        node=dates_leaf,
        additional_instruction="Use common-sense parsing of month names or numerals. Accept if both dates clearly indicate year 2026 and the months align with the specified seasonal window."
    )

    # Submission deadline present; if labeled estimate, it is 3–8 months before start (critical)
    sub_leaf = evaluator.add_leaf(
        id=f"{item_id_prefix}_Conference_Submission_Deadline",
        desc="A submission deadline date is provided. If the answer labels it as an estimate, the estimated deadline is 3–8 months before the conference start date (per constraints).",
        parent=parent_node,
        critical=True
    )
    sub_deadline = conf.submission_deadline or ""
    is_est = bool(conf.submission_deadline_is_estimate)
    await evaluator.verify(
        claim=(
            f"A submission deadline is provided ('{sub_deadline}'). "
            f"If labeled as an estimate ({'yes' if is_est else 'no'}), it should be roughly 3–8 months before the start date '{sd}'."
        ),
        node=sub_leaf,
        additional_instruction="Judge correct if a submission deadline is present. If and only if the answer explicitly marks it as an estimate, also check it is plausibly 3–8 months before the start date (approximate reasoning is fine)."
    )

    # Verification URL provided and supports dates/location (critical)
    verif_leaf = evaluator.add_leaf(
        id=f"{item_id_prefix}_Conference_Verification_URL",
        desc="A verification URL is provided for the conference dates/location (official conference site or reputable host organization page).",
        parent=parent_node,
        critical=True
    )
    # Combine existence and content support in one claim
    await evaluator.verify(
        claim=(
            f"A verification URL is provided, and the page shows that the '{conf.name or ''}' conference "
            f"takes place in '{conf.location or ''}' on '{sd}' to '{ed}' in 2026."
        ),
        node=verif_leaf,
        sources=conf.verification_url,
        additional_instruction=(
            "If no URL is provided in the answer, mark incorrect. If a URL is provided, "
            "check that the page supports the stated dates and location for the named conference."
        )
    )


async def verify_conferences_2026(
    evaluator: Evaluator,
    root_node
) -> None:
    # Parent node for conferences (critical)
    conf_parent = evaluator.add_parallel(
        id="Conference_Presentations_2026",
        desc="Provide TWO major education research conferences in 2026 (one spring, one fall) with required details and a verification URL for each.",
        parent=root_node,
        critical=True
    )

    # Retrieve the already-extracted conference info from evaluator's recorded extractions (we'll re-extract here explicitly)
    # For clarity, we expect the caller to provide the ConferencesExtraction object.
    # We'll fetch it from a custom info slot if provided, else do nothing here.
    # But in our flow, we'll pass the extracted object directly into this helper, so not used.
    pass


async def verify_journal_plan(
    evaluator: Evaluator,
    parent_node,
    journal: JournalPlanExtraction
) -> None:
    # Journal name provided (critical within this (non-critical) section)
    evaluator.add_custom_node(
        result=bool(journal and journal.journal_name and journal.journal_name.strip()),
        id="Journal_Name",
        desc="Journal name is provided.",
        parent=parent_node,
        critical=True
    )

    # Journal verification URL (critical): page supports the journal identity
    j_url_leaf = evaluator.add_leaf(
        id="Journal_Verification_URL",
        desc="A URL is provided verifying the journal (official site or reputable journal profile).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"A verification URL is provided and the page corresponds to the journal '{journal.journal_name or ''}'.",
        node=j_url_leaf,
        sources=journal.verification_url,
        additional_instruction="If no URL is provided in the answer, judge incorrect. If a URL is provided, verify the page is about the specified journal."
    )

    # Quality indicator (non-critical): provided OR explicitly 'not available'
    qi_str = (journal.quality_indicator or "").strip().lower()
    has_quality = bool(journal.quality_indicator and journal.quality_indicator.strip())
    explicitly_not_available = qi_str in {"not available", "not found", "n/a", "none"}
    evaluator.add_custom_node(
        result=has_quality or explicitly_not_available,
        id="Journal_Quality_Indicator",
        desc="Provides at least one quality indicator OR explicitly states that a quality indicator was not found/available (e.g., impact factor, quartile, indexing).",
        parent=parent_node,
        critical=False
    )

    # Proposed submission timeline (critical): must include explicit submission timing + durations + target before Dec 1–15, 2026
    timeline_leaf = evaluator.add_leaf(
        id="Proposed_Submission_Timeline",
        desc="Provides a dated/dated-range plan that: (1) includes a submission timing, (2) incorporates 12–14 weeks to initial decision and 3–6 weeks for revision, and (3) explicitly targets completion or at least R&R status before December 1–15, 2026.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The plan includes an explicit submission timing ('{journal.proposed_submission_timing or ''}'), "
            f"accounts for 12–14 weeks to initial decision and 3–6 weeks for revisions, "
            f"and aims to complete or reach R&R before early December 2026 (Dec 1–15)."
        ),
        node=timeline_leaf,
        additional_instruction=(
            "Judge correct only if the answer clearly includes a submission date/range, explicitly mentions both the 12–14 week initial decision window and a 3–6 week revision window, "
            "and explicitly aims to complete or at least reach R&R before Dec 1–15, 2026."
        )
    )


async def verify_phd_program(
    evaluator: Evaluator,
    parent_node,
    phd: PhDProgramExtraction
) -> None:
    # University name (critical)
    evaluator.add_custom_node(
        result=bool(phd and phd.university_name and phd.university_name.strip()),
        id="University_Name",
        desc="University name is provided.",
        parent=parent_node,
        critical=True
    )

    # Program name (critical)
    evaluator.add_custom_node(
        result=bool(phd and phd.program_name and phd.program_name.strip()),
        id="Program_Name",
        desc="Program/department name is provided.",
        parent=parent_node,
        critical=True
    )

    # Program field relevance (critical) - verify via program/admissions URLs
    field_leaf = evaluator.add_leaf(
        id="Program_Field_Relevance",
        desc="Program is in education or a related field, supported by program title/description or source text.",
        parent=parent_node,
        critical=True
    )
    program_related_urls = (phd.program_urls or []) + (phd.admissions_urls or [])
    await evaluator.verify(
        claim=(
            f"The program '{phd.program_name or ''}' at {phd.university_name or ''} is in education or a closely related field "
            f"(e.g., educational psychology, learning sciences)."
        ),
        node=field_leaf,
        sources=program_related_urls,
        additional_instruction="Allow reasonable related fields if clearly connected to education in the page content."
    )

    # Research university indicator (critical)
    rsch_leaf = evaluator.add_leaf(
        id="Research_University_Indicator",
        desc="Includes a verifiable indicator the university is a research university (e.g., Carnegie R1/R2 classification or equivalent) with a supporting URL.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The university '{phd.university_name or ''}' is a research university (e.g., Carnegie R1/R2 or equivalent)."
        ),
        node=rsch_leaf,
        sources=phd.research_indicator_url,
        additional_instruction="If no research-indicator URL is provided in the answer, mark incorrect. If provided, verify that the page explicitly indicates research university status (Carnegie classification or similar)."
    )

    # Minimum GPA requirement (critical)
    gpa_leaf = evaluator.add_leaf(
        id="Minimum_GPA_Requirement",
        desc="Minimum GPA requirement is stated for the identified program and is source-backed.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum GPA requirement for the program is '{phd.minimum_gpa or ''}'.",
        node=gpa_leaf,
        sources=phd.admissions_urls,
        additional_instruction="If no admissions URL is provided in the answer, mark incorrect. Verify the page states the minimum GPA requirement equal to or consistent with the claim."
    )

    # Recommendation letters required (critical)
    rec_leaf = evaluator.add_leaf(
        id="Recommendation_Letters_Required",
        desc="Number of required recommendation letters is stated for the identified program and is source-backed.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program requires '{phd.letters_required or ''}' recommendation letters.",
        node=rec_leaf,
        sources=phd.admissions_urls,
        additional_instruction="If no admissions URL is provided in the answer, mark incorrect. Verify the page shows the same number of required recommendation letters."
    )

    # Application deadline Fall 2027 (critical)
    deadline_leaf = evaluator.add_leaf(
        id="Application_Deadline_Fall_2027",
        desc="Application deadline for Fall 2027 admission is provided and is source-backed.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The application deadline for Fall 2027 admission is '{phd.application_deadline_fall_2027 or ''}'.",
        node=deadline_leaf,
        sources=phd.admissions_urls,
        additional_instruction="If no admissions URL is provided in the answer, mark incorrect. Verify that the page shows the same Fall 2027 application deadline."
    )

    # Recommendation request timeline 2–3 months before deadline (critical)
    rec_time_leaf = evaluator.add_leaf(
        id="Recommendation_Request_Timeline",
        desc="Provides a request timeline that is 2–3 months before the stated application deadline (per constraints).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The recommended letter request timeline '{phd.recommendation_request_timeline or ''}' is approximately 2–3 months before the application deadline '{phd.application_deadline_fall_2027 or ''}'."
        ),
        node=rec_time_leaf,
        additional_instruction="Judge correct if the timing is plausibly 2–3 months before the given deadline, even if described as a month range rather than an exact date."
    )

    # Program verification URLs presence (critical)
    evaluator.add_custom_node(
        result=(len(phd.program_urls or []) >= 1 and len(phd.admissions_urls or []) >= 1),
        id="Program_Verification_URLs",
        desc="Provides at least one verification URL for program identity and at least one URL for admissions requirements/deadline (can be the same page if it contains both).",
        parent=parent_node,
        critical=True
    )


async def verify_fellowship(
    evaluator: Evaluator,
    parent_node,
    f: FellowshipExtraction
) -> None:
    # Fellowship name provided (critical)
    evaluator.add_custom_node(
        result=bool(f and f.fellowship_name and f.fellowship_name.strip()),
        id="Fellowship_Name",
        desc="Fellowship program name is provided.",
        parent=parent_node,
        critical=True
    )

    # Explicitly postdoctoral (critical)
    is_postdoc_leaf = evaluator.add_leaf(
        id="Fellowship_Is_Postdoctoral",
        desc="The opportunity is explicitly a postdoctoral fellowship (as stated by the program/source).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The fellowship '{f.fellowship_name or ''}' is explicitly a postdoctoral fellowship.",
        node=is_postdoc_leaf,
        sources=f.url,
        additional_instruction="If no URL is provided in the answer, mark incorrect. Verify that the page explicitly states this is a postdoctoral fellowship."
    )

    # Relevance to education (critical)
    rel_leaf = evaluator.add_leaf(
        id="Fellowship_Relevance_To_Education",
        desc="Provides source-backed evidence the fellowship supports education or education-related research/training (or an explicitly compatible broader area).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The fellowship supports education or education-related research/training (or a clearly compatible broader area).",
        node=rel_leaf,
        sources=f.url,
        additional_instruction="Verify on the page that the fellowship's focus or eligibility aligns with education or education-related research/training."
    )

    # Eligibility requirements (critical)
    elig_leaf = evaluator.add_leaf(
        id="Eligibility_Requirements",
        desc="Eligibility requirements are provided (including PhD completion/near-completion, if applicable) and are consistent with the source.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The eligibility requirements include: {f.eligibility or ''}",
        node=elig_leaf,
        sources=f.url,
        additional_instruction="Verify that the page contains eligibility requirements consistent with the stated text."
    )

    # Application deadline (critical)
    f_deadline_leaf = evaluator.add_leaf(
        id="Application_Deadline",
        desc="Application deadline is provided and source-backed.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The application deadline is '{f.application_deadline or ''}'.",
        node=f_deadline_leaf,
        sources=f.url,
        additional_instruction="Verify the page shows the same application deadline."
    )

    # Funding information (non-critical)
    fund_leaf = evaluator.add_leaf(
        id="Funding_Information",
        desc="Funding information is provided if available in the source.",
        parent=parent_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The fellowship provides funding information such as '{f.funding_info or ''}'.",
        node=fund_leaf,
        sources=f.url,
        additional_instruction="Judge correct if the page presents any funding information consistent with the answer. If the answer does not include funding info and the page lacks it, this may be marked incorrect; however, this item is non-critical."
    )

    # Fellowship verification URL (critical)
    f_url_leaf = evaluator.add_leaf(
        id="Fellowship_Verification_URL",
        desc="A URL is provided that verifies the fellowship details (official program page preferred).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"A verification URL is provided and the page corresponds to the fellowship '{f.fellowship_name or ''}'.",
        node=f_url_leaf,
        sources=f.url,
        additional_instruction="If no URL is provided in the answer, mark incorrect. If a URL is provided, verify that the page is about the stated fellowship program."
    )


async def verify_overall_timeline(
    evaluator: Evaluator,
    parent_node,
    conferences: ConferencesExtraction,
    journal: JournalPlanExtraction,
    phd: PhDProgramExtraction,
    fellowship: FellowshipExtraction
) -> None:
    # Chronological coordination explanation (critical)
    chron_leaf = evaluator.add_leaf(
        id="Chronological_Coordination_Explanation",
        desc="Provides an integrated chronological plan that explicitly references all four goal areas (two conferences, journal plan, PhD application, fellowship) and their key deadlines/windows.",
        parent=parent_node,
        critical=True
    )
    spring = conferences.spring or ConferenceItem()
    fall = conferences.fall or ConferenceItem()
    await evaluator.verify(
        claim=(
            "The answer includes an integrated chronological plan that explicitly references: "
            f"(1) Spring 2026 conference '{spring.name or ''}' with its submission deadline '{spring.submission_deadline or ''}', "
            f"(2) Fall 2026 conference '{fall.name or ''}' with its submission deadline '{fall.submission_deadline or ''}', "
            f"(3) the journal submission plan with submission timing '{journal.proposed_submission_timing or ''}', and "
            f"(4) the PhD application (deadline '{phd.application_deadline_fall_2027 or ''}') and the fellowship timeline."
        ),
        node=chron_leaf,
        additional_instruction="Judge correct only if the narrative clearly coordinates all four areas and explicitly mentions their key deadlines/windows in a chronological way."
    )

    # Internal consistency check (critical)
    consistency_leaf = evaluator.add_leaf(
        id="Internal_Consistency_Check",
        desc="Timeline is internally consistent: journal review + revision windows fit between stated submission timing and December 1–15, 2026 PhD deadlines; recommendation-request timing is 2–3 months before the stated PhD deadline; conference deadlines precede their conference dates.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The planned timeline is internally consistent: "
            f"the journal submission timing '{journal.proposed_submission_timing or ''}' plus 12–14 weeks to initial decision and 3–6 weeks for revisions fits before early-December 2026 (Dec 1–15); "
            f"the recommendation request window '{phd.recommendation_request_timeline or ''}' is about 2–3 months before the PhD deadline '{phd.application_deadline_fall_2027 or ''}'; "
            f"and conference submission deadlines ('{spring.submission_deadline or ''}' and '{fall.submission_deadline or ''}') precede their respective conference dates ('{spring.start_date or ''}' and '{fall.start_date or ''}')."
        ),
        node=consistency_leaf,
        additional_instruction="Use approximate calendar reasoning. Fail if any of these relationships obviously do not hold or are missing."
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
    Build the verification tree and run evaluation for the 2026–2027 academic planning task.
    """
    # Initialize evaluator with a parallel root
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

    # Extract all sections in parallel
    conferences_task = evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_extraction"
    )
    journal_task = evaluator.extract(
        prompt=prompt_extract_journal_plan(),
        template_class=JournalPlanExtraction,
        extraction_name="journal_plan_extraction"
    )
    phd_task = evaluator.extract(
        prompt=prompt_extract_phd_program(),
        template_class=PhDProgramExtraction,
        extraction_name="phd_program_extraction"
    )
    fellowship_task = evaluator.extract(
        prompt=prompt_extract_fellowship(),
        template_class=FellowshipExtraction,
        extraction_name="fellowship_extraction"
    )

    conferences, journal, phd, fellowship = await asyncio.gather(
        conferences_task, journal_task, phd_task, fellowship_task
    )

    # -------------------- Build Top-Level Nodes --------------------------- #
    # Conferences (critical)
    conferences_node = evaluator.add_parallel(
        id="Conference_Presentations_2026",
        desc="Provide TWO major education research conferences in 2026 (one spring, one fall) with required details and a verification URL for each.",
        parent=root,
        critical=True
    )

    # Spring conference item (critical)
    spring_node = evaluator.add_parallel(
        id="Spring_Conference_Item",
        desc="Spring 2026 conference details.",
        parent=conferences_node,
        critical=True
    )
    await verify_conference_item(evaluator, spring_node, conferences.spring or ConferenceItem(), "spring")

    # Fall conference item (critical)
    fall_node = evaluator.add_parallel(
        id="Fall_Conference_Item",
        desc="Fall 2026 conference details.",
        parent=conferences_node,
        critical=True
    )
    await verify_conference_item(evaluator, fall_node, conferences.fall or ConferenceItem(), "fall")

    # Journal submission plan (set non-critical to allow optional quality indicator)
    journal_node = evaluator.add_parallel(
        id="Journal_Submission_Plan",
        desc="Identify a journal and provide a submission timeline that accounts for review/revision timing and PhD deadlines, including a verification URL.",
        parent=root,
        critical=False
    )
    await verify_journal_plan(evaluator, journal_node, journal or JournalPlanExtraction())

    # PhD program application (critical)
    phd_node = evaluator.add_parallel(
        id="PhD_Program_Application_Fall_2027",
        desc="Identify one PhD program and provide required admissions attributes and verification URLs.",
        parent=root,
        critical=True
    )
    await verify_phd_program(evaluator, phd_node, phd or PhDProgramExtraction())

    # Postdoctoral fellowship opportunity (set non-critical to allow optional funding info)
    fellowship_node = evaluator.add_parallel(
        id="Postdoctoral_Fellowship_Opportunity",
        desc="Identify one relevant postdoctoral fellowship and provide required attributes and a verification URL.",
        parent=root,
        critical=False
    )
    await verify_fellowship(evaluator, fellowship_node, fellowship or FellowshipExtraction())

    # Overall timeline coordination (critical)
    overall_node = evaluator.add_parallel(
        id="Overall_Timeline_Coordination",
        desc="Demonstrate how the deadlines and requirements are coordinated into a feasible 2026–2027 timeline.",
        parent=root,
        critical=True
    )
    await verify_overall_timeline(evaluator, overall_node, conferences, journal, phd, fellowship)

    # Return summary
    return evaluator.get_summary()