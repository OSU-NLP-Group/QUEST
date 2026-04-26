import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "ai_conf_2026_constraints"
TASK_DESCRIPTION = """
A PhD student in artificial intelligence is preparing to submit their machine learning research paper for publication at a major international conference in 2026. The student has the following specific requirements and constraints:

Timing Requirements:
- The paper submission deadline must be between January 1 and February 28, 2026, to align with their research completion timeline
- The conference must take place between June 1 and August 31, 2026, to fit within their academic calendar
- Early bird registration must be available at least 8 weeks before the conference starts

Budget and Registration:
- As a student, they require a student early bird registration fee of $500 USD or less
- They understand that at least one co-author must register at the full author registration rate

Submission and Review Requirements:
- The conference must accept full research papers of at least 8 pages in double-column format
- The peer review process must be double-blind to ensure fairness
- Supplementary materials submission must be allowed to include additional experimental results
- The abstract submission must allow for at least 250 words to adequately summarize their work
- An author rebuttal or response phase must be available during the review process to address reviewer concerns

Publication Quality:
- The conference proceedings must be indexed in IEEE Xplore or Scopus for visibility and citation purposes
- The conference must be a recognized major conference in the AI/Machine Learning field
- Proceedings must be published in an established, reputable conference series

Location and Logistics:
- The conference venue must be in the United States or Europe for accessibility
- The venue must meet accessibility standards including wheelchair-accessible entrances and exits
- Hotel accommodations at negotiated conference rates must be available

Support and Services:
- A student volunteer program with defined benefits must be offered
- Visa support letters must be available for international attendees
- The conference must accommodate dietary restrictions for meals and receptions

Presentation Requirements:
- Both oral presentation and poster presentation options must be available
- If presenting a poster, the required size must not exceed 48 inches in any dimension
- Oral presentations must have clearly defined time allocations

Identify one specific major AI/ML conference in 2026 that satisfies all of these requirements. For each requirement, provide the specific information that demonstrates compliance and include the URL reference where this information can be verified.
"""


# =========================
# Extraction Models
# =========================
class ConferenceIdentification(BaseModel):
    name: Optional[str] = None
    official_url: Optional[str] = None
    major_evidence_urls: List[str] = Field(default_factory=list)
    year_evidence_urls: List[str] = Field(default_factory=list)


class DatesAndRegistration(BaseModel):
    submission_deadline: Optional[str] = None
    submission_urls: List[str] = Field(default_factory=list)

    conf_start_date: Optional[str] = None
    conf_end_date: Optional[str] = None
    conf_dates_urls: List[str] = Field(default_factory=list)

    early_bird_deadline: Optional[str] = None
    early_bird_urls: List[str] = Field(default_factory=list)

    student_early_bird_fee_usd: Optional[str] = None
    fee_urls: List[str] = Field(default_factory=list)

    fee_structure_urls: List[str] = Field(default_factory=list)
    author_registration_required_urls: List[str] = Field(default_factory=list)


class SubmissionAndReview(BaseModel):
    full_papers_min_pages: Optional[str] = None
    full_papers_format_urls: List[str] = Field(default_factory=list)  # Must include double-column requirement

    double_blind_urls: List[str] = Field(default_factory=list)
    supplementary_allowed_urls: List[str] = Field(default_factory=list)

    abstract_word_limit: Optional[str] = None
    abstract_urls: List[str] = Field(default_factory=list)

    rebuttal_urls: List[str] = Field(default_factory=list)


class PublicationQuality(BaseModel):
    indexing_target: Optional[str] = None  # e.g., "IEEE Xplore" or "Scopus"
    indexing_urls: List[str] = Field(default_factory=list)

    series_name: Optional[str] = None
    series_urls: List[str] = Field(default_factory=list)


class LocationAndLogistics(BaseModel):
    venue_location: Optional[str] = None  # e.g., "San Francisco, USA"
    venue_urls: List[str] = Field(default_factory=list)

    accessibility_urls: List[str] = Field(default_factory=list)  # wheelchair-accessible confirmation
    hotel_rates_urls: List[str] = Field(default_factory=list)  # negotiated conference rates / room block


class SupportAndServices(BaseModel):
    student_volunteer_urls: List[str] = Field(default_factory=list)
    visa_support_urls: List[str] = Field(default_factory=list)
    dietary_urls: List[str] = Field(default_factory=list)


class PresentationRequirements(BaseModel):
    oral_poster_urls: List[str] = Field(default_factory=list)
    poster_max_size_in: Optional[str] = None  # e.g., "48 inches x 36 inches"
    poster_urls: List[str] = Field(default_factory=list)
    oral_time_alloc_urls: List[str] = Field(default_factory=list)


class ConferenceExtraction(BaseModel):
    identification: Optional[ConferenceIdentification] = None
    dates_registration: Optional[DatesAndRegistration] = None
    submission_review: Optional[SubmissionAndReview] = None
    publication_quality: Optional[PublicationQuality] = None
    location_logistics: Optional[LocationAndLogistics] = None
    support_services: Optional[SupportAndServices] = None
    presentation: Optional[PresentationRequirements] = None


# =========================
# Extraction Prompt
# =========================
def prompt_extract_conference() -> str:
    return """
    Extract structured information for one specific major AI/ML conference in 2026 from the answer. 
    Return a JSON that matches exactly the following fields and nesting. 
    IMPORTANT:
    - Extract only URLs that are explicitly present in the answer. Do not invent or infer URLs.
    - If a required URL is missing in the answer for a specific requirement, return an empty array [] for that field.
    - Keep dates and numeric values as strings exactly as they appear in the answer (do not normalize to a specific format).
    - The "official_url" should be the conference's official website or an official organizer page referenced in the answer.

    JSON schema (fill as much as the answer provides):
    {
      "identification": {
        "name": string | null,
        "official_url": string | null,
        "major_evidence_urls": [string],
        "year_evidence_urls": [string]
      },
      "dates_registration": {
        "submission_deadline": string | null,
        "submission_urls": [string],
        "conf_start_date": string | null,
        "conf_end_date": string | null,
        "conf_dates_urls": [string],
        "early_bird_deadline": string | null,
        "early_bird_urls": [string],
        "student_early_bird_fee_usd": string | null,
        "fee_urls": [string],
        "fee_structure_urls": [string],
        "author_registration_required_urls": [string]
      },
      "submission_review": {
        "full_papers_min_pages": string | null,
        "full_papers_format_urls": [string],
        "double_blind_urls": [string],
        "supplementary_allowed_urls": [string],
        "abstract_word_limit": string | null,
        "abstract_urls": [string],
        "rebuttal_urls": [string]
      },
      "publication_quality": {
        "indexing_target": string | null,
        "indexing_urls": [string],
        "series_name": string | null,
        "series_urls": [string]
      },
      "location_logistics": {
        "venue_location": string | null,
        "venue_urls": [string],
        "accessibility_urls": [string],
        "hotel_rates_urls": [string]
      },
      "support_services": {
        "student_volunteer_urls": [string],
        "visa_support_urls": [string],
        "dietary_urls": [string]
      },
      "presentation": {
        "oral_poster_urls": [string],
        "poster_max_size_in": string | null,
        "poster_urls": [string],
        "oral_time_alloc_urls": [string]
      }
    }

    Notes:
    - "major_evidence_urls": links that indicate the conference is recognized as a major international conference in AI/ML (or core subfields), e.g., official pages, rankings, Wikipedia descriptions, etc., as provided in the answer.
    - "year_evidence_urls": links that demonstrate the conference is in 2026 (e.g., official site indicating 2026 edition or dates).
    - "full_papers_format_urls": links showing acceptance of full research papers of at least 8 pages in double-column format.
    - "indexing_urls": links showing proceedings indexed in IEEE Xplore or Scopus.
    - "series_urls": links showing proceedings published in an established reputable conference series (e.g., LNCS, IEEE CPS, ACM ICPS, etc.).
    - "accessibility_urls": links or venue pages explicitly stating wheelchair-accessible entrances and exits.
    - "hotel_rates_urls": links indicating negotiated conference hotel rates or room blocks.
    - "oral_poster_urls": links indicating both oral and poster presentation options exist.
    - "poster_urls": links indicating poster size rules and maximum dimensions.
    - "oral_time_alloc_urls": links indicating oral presentation time allocations are defined.
    """


# =========================
# Helper Functions
# =========================
def _safe(s: Optional[str]) -> str:
    return s or ""

def _urls(lst: Optional[List[str]]) -> List[str]:
    return lst or []

def _merge_sources(*url_lists: List[str], fallback: Optional[str] = None) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str) and u.strip() and u not in merged:
                merged.append(u)
    if fallback and fallback.strip() and fallback not in merged:
        merged.append(fallback)
    return merged

def _add_sources_gate(evaluator: Evaluator, parent, node_id: str, desc: str, urls: List[str]) -> Any:
    return evaluator.add_custom_node(
        result=(len(urls) > 0),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True
    )


# =========================
# Verification Builders
# =========================
async def build_conference_identification(
    evaluator: Evaluator,
    parent_node,
    ext: ConferenceExtraction,
) -> None:
    ident = ext.identification or ConferenceIdentification()
    dates_reg = ext.dates_registration or DatesAndRegistration()

    ci_node = evaluator.add_parallel(
        id="Conference_Identification",
        desc="Identify the conference and provide an official reference page.",
        parent=parent_node,
        critical=True
    )

    # Conference_Name_Provided (existence)
    evaluator.add_custom_node(
        result=bool(ident.name and ident.name.strip()),
        id="Conference_Name_Provided",
        desc="Provides the specific conference name/edition being proposed.",
        parent=ci_node,
        critical=True
    )

    # Official URL provided gate
    official_url_provided = evaluator.add_custom_node(
        result=bool(ident.official_url and ident.official_url.strip()),
        id="Official_Conference_URL_Provided",
        desc="Provides an official conference website (or equivalent official organizer page) URL is present in the answer.",
        parent=ci_node,
        critical=True
    )

    # Official_Conference_URL (verify that the provided URL is indeed official)
    official_leaf = evaluator.add_leaf(
        id="Official_Conference_URL",
        desc="Provides an official conference website (or equivalent official organizer page) URL.",
        parent=ci_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The URL is the official website or official organizer page for the { _safe(ident.name) } 2026 conference.",
        node=official_leaf,
        sources=ident.official_url,
        additional_instruction="Verify that the page is clearly the official conference site (or an official organizer page), not a third-party aggregator."
    )

    # Conference_Is_2026 (verify via year_evidence_urls or conf_dates)
    year_sources = _merge_sources(_urls(ident.year_evidence_urls), _urls(dates_reg.conf_dates_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, ci_node,
        "Conference_Is_2026_Sources_Provided",
        "Source URLs are provided for confirming the conference is in calendar year 2026.",
        year_sources
    )
    year_leaf = evaluator.add_leaf(
        id="Conference_Is_2026",
        desc="Provides evidence (with URL) that the conference is in calendar year 2026.",
        parent=ci_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The referenced page(s) indicate that { _safe(ident.name) } takes place in 2026.",
        node=year_leaf,
        sources=year_sources,
        additional_instruction="Look for edition labeling or official dates to confirm year 2026."
    )

    # Conference_Is_Major_AI_ML
    major_sources = _merge_sources(_urls(ident.major_evidence_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, ci_node,
        "Conference_Is_Major_AI_ML_Sources_Provided",
        "Source URLs are provided for conference recognition as major AI/ML conference.",
        major_sources
    )
    major_leaf = evaluator.add_leaf(
        id="Conference_Is_Major_AI_ML",
        desc="Provides evidence (with URL) that it is a recognized major international conference in AI/Machine Learning (or a core subfield like NLP/CV/ML) rather than a workshop/regional event.",
        parent=ci_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) demonstrate that { _safe(ident.name) } is a recognized major international conference in AI/ML (or core subfield).",
        node=major_leaf,
        sources=major_sources,
        additional_instruction="Accept reputable evidence such as official statements, widely recognized rankings, Wikipedia descriptions, or organizer claims indicating premier status."
    )


async def build_constraint_checks(
    evaluator: Evaluator,
    parent_node,
    ext: ConferenceExtraction,
) -> None:
    ident = ext.identification or ConferenceIdentification()
    dr = ext.dates_registration or DatesAndRegistration()
    sr = ext.submission_review or SubmissionAndReview()
    pq = ext.publication_quality or PublicationQuality()
    ll = ext.location_logistics or LocationAndLogistics()
    ss = ext.support_services or SupportAndServices()
    pr = ext.presentation or PresentationRequirements()

    cc_node = evaluator.add_parallel(
        id="Constraint_Compliance_Checks",
        desc="Each explicit constraint is satisfied and backed by specific information + a URL reference.",
        parent=parent_node,
        critical=True
    )

    # Submission_Deadline_Window
    sub_sources = _merge_sources(_urls(dr.submission_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Submission_Deadline_Window_Sources_Provided",
        "Source URLs provided for submission deadline window.",
        sub_sources
    )
    sub_leaf = evaluator.add_leaf(
        id="Submission_Deadline_Window",
        desc="Provides the paper submission deadline date and a URL showing it is between Jan 1 and Feb 28, 2026.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The submission deadline ({_safe(dr.submission_deadline)}) is between January 1 and February 28, 2026 (inclusive).",
        node=sub_leaf,
        sources=sub_sources,
        additional_instruction="Confirm the deadline is in the specified window; interpret date formats flexibly."
    )

    # Conference_Date_Window
    conf_dates_sources = _merge_sources(_urls(dr.conf_dates_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Conference_Date_Window_Sources_Provided",
        "Source URLs provided for conference date window.",
        conf_dates_sources
    )
    conf_dates_leaf = evaluator.add_leaf(
        id="Conference_Date_Window",
        desc="Provides the main conference start/end dates and a URL showing they fall between Jun 1 and Aug 31, 2026.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The conference dates ({_safe(dr.conf_start_date)} to {_safe(dr.conf_end_date)}) fall entirely between June 1 and August 31, 2026.",
        node=conf_dates_leaf,
        sources=conf_dates_sources,
        additional_instruction="Use the official main conference dates (not workshops) to judge the window."
    )

    # Early_Bird_At_Least_8_Weeks_Before_Start
    eb_sources = _merge_sources(_urls(dr.early_bird_urls), _urls(dr.conf_dates_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Early_Bird_At_Least_8_Weeks_Before_Start_Sources_Provided",
        "Source URLs provided for early-bird registration timing and conference start date.",
        eb_sources
    )
    eb_leaf = evaluator.add_leaf(
        id="Early_Bird_At_Least_8_Weeks_Before_Start",
        desc="Provides the early-bird registration deadline and conference start date (with URL evidence) showing the deadline is ≥ 8 weeks before the start date.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The early-bird registration deadline ({_safe(dr.early_bird_deadline)}) is at least 8 weeks (56 days) before the conference start date ({_safe(dr.conf_start_date)}).",
        node=eb_leaf,
        sources=eb_sources,
        additional_instruction="Compute the difference between the early-bird deadline and the start date; consider inclusive counting; at least 56 days difference qualifies."
    )

    # Student_Early_Bird_Fee_Cap
    fee_sources = _merge_sources(_urls(dr.fee_urls), _urls(dr.fee_structure_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Student_Early_Bird_Fee_Cap_Sources_Provided",
        "Source URLs provided for student early-bird fee.",
        fee_sources
    )
    fee_leaf = evaluator.add_leaf(
        id="Student_Early_Bird_Fee_Cap",
        desc="Provides the student early-bird registration fee (with URL evidence) showing it is ≤ $500 USD.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The student early-bird registration fee is {_safe(dr.student_early_bird_fee_usd)} USD and is ≤ $500.",
        node=fee_leaf,
        sources=fee_sources,
        additional_instruction="Confirm the amount is clearly for student early-bird; if fee shown is in other currency, convert only if page provides USD equivalence; otherwise judge against stated USD value if present."
    )

    # Author_Registration_Required
    arr_sources = _merge_sources(_urls(dr.author_registration_required_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Author_Registration_Required_Sources_Provided",
        "Source URLs provided for author registration requirement.",
        arr_sources
    )
    arr_leaf = evaluator.add_leaf(
        id="Author_Registration_Required",
        desc="Provides policy text (with URL evidence) that at least one author/co-author registration is required per accepted paper.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one author/co-author must register at the full author rate for each accepted paper.",
        node=arr_leaf,
        sources=arr_sources,
        additional_instruction="Look for 'author registration required' policies; phrasing may vary."
    )

    # Fee_Structure_Clearly_Specified_With_Early_Bird
    fs_sources = _merge_sources(_urls(dr.fee_structure_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Fee_Structure_Clearly_Specified_With_Early_Bird_Sources_Provided",
        "Source URLs provided for fee structure and early-bird tier.",
        fs_sources
    )
    fs_leaf = evaluator.add_leaf(
        id="Fee_Structure_Clearly_Specified_With_Early_Bird",
        desc="Provides a URL showing the registration fee structure includes an early-bird pricing tier (i.e., fees and deadlines are clearly specified).",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The registration fee structure includes an explicitly defined early-bird pricing tier and associated deadlines.",
        node=fs_leaf,
        sources=fs_sources,
        additional_instruction="The schedule must show early-bird tier; accept clear table/listing."
    )

    # Full_Papers_Min_8_Pages_Double_Column
    fpf_sources = _merge_sources(_urls(sr.full_papers_format_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Full_Papers_Min_8_Pages_Double_Column_Sources_Provided",
        "Source URLs provided for full paper length and double-column format.",
        fpf_sources
    )
    fpf_leaf = evaluator.add_leaf(
        id="Full_Papers_Min_8_Pages_Double_Column",
        desc="Provides submission/formatting rules (with URL evidence) showing full research papers of at least 8 pages in double-column format are accepted.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Full research papers of at least {_safe(sr.full_papers_min_pages)} pages in double-column format are accepted.",
        node=fpf_leaf,
        sources=fpf_sources,
        additional_instruction="The guidelines must clearly indicate both minimum page length (≥8) and double-column format."
    )

    # Double_Blind_Peer_Review
    db_sources = _merge_sources(_urls(sr.double_blind_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Double_Blind_Peer_Review_Sources_Provided",
        "Source URLs provided for double-blind peer review policy.",
        db_sources
    )
    db_leaf = evaluator.add_leaf(
        id="Double_Blind_Peer_Review",
        desc="Provides review policy/process text (with URL evidence) stating the peer review is double-blind.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The peer review process is double-blind.",
        node=db_leaf,
        sources=db_sources,
        additional_instruction="Look for explicit 'double-blind' wording in review policy."
    )

    # Supplementary_Materials_Allowed
    sm_sources = _merge_sources(_urls(sr.supplementary_allowed_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Supplementary_Materials_Allowed_Sources_Provided",
        "Source URLs provided for supplementary materials allowance.",
        sm_sources
    )
    sm_leaf = evaluator.add_leaf(
        id="Supplementary_Materials_Allowed",
        desc="Provides submission guidelines (with URL evidence) that supplementary materials submission is allowed.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Supplementary materials submission is allowed.",
        node=sm_leaf,
        sources=sm_sources,
        additional_instruction="Look for sections indicating supplementary material policies."
    )

    # Abstract_Allows_At_Least_250_Words
    abs_sources = _merge_sources(_urls(sr.abstract_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Abstract_Allows_At_Least_250_Words_Sources_Provided",
        "Source URLs provided for abstract word limit.",
        abs_sources
    )
    abs_leaf = evaluator.add_leaf(
        id="Abstract_Allows_At_Least_250_Words",
        desc="Provides abstract length limit (with URL evidence) showing at least 250 words are allowed (or an equivalent limit that clearly permits ≥250 words).",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The abstract submission allows at least 250 words; the stated limit is {_safe(sr.abstract_word_limit)} words or equivalent that clearly permits ≥250 words.",
        node=abs_leaf,
        sources=abs_sources,
        additional_instruction="If limit is in characters or lines, judge whether it reasonably allows ≥250 words."
    )

    # Rebuttal_Or_Response_Phase
    reb_sources = _merge_sources(_urls(sr.rebuttal_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Rebuttal_Or_Response_Phase_Sources_Provided",
        "Source URLs provided for rebuttal/response phase.",
        reb_sources
    )
    reb_leaf = evaluator.add_leaf(
        id="Rebuttal_Or_Response_Phase",
        desc="Provides review timeline/process (with URL evidence) showing an author rebuttal/response phase exists.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="An author rebuttal or response phase exists during the review process.",
        node=reb_leaf,
        sources=reb_sources,
        additional_instruction="Accept explicit mention of rebuttal/response window in timeline or process page."
    )

    # Proceedings_Indexed_IEEE_Xplore_Or_Scopus
    idx_sources = _merge_sources(_urls(pq.indexing_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Proceedings_Indexed_IEEE_Xplore_Or_Scopus_Sources_Provided",
        "Source URLs provided for indexing in IEEE Xplore or Scopus.",
        idx_sources
    )
    idx_leaf = evaluator.add_leaf(
        id="Proceedings_Indexed_IEEE_Xplore_Or_Scopus",
        desc="Provides URL evidence that the proceedings are indexed in IEEE Xplore or Scopus (must be one of these, not a different database substitute).",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The proceedings are indexed in {_safe(pq.indexing_target)} (IEEE Xplore or Scopus).",
        node=idx_leaf,
        sources=idx_sources,
        additional_instruction="Confirm that indexing includes IEEE Xplore or Scopus explicitly; other databases alone do not satisfy this requirement."
    )

    # Proceedings_In_Established_Conference_Series
    series_sources = _merge_sources(_urls(pq.series_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Proceedings_In_Established_Conference_Series_Sources_Provided",
        "Source URLs provided for proceedings series.",
        series_sources
    )
    series_leaf = evaluator.add_leaf(
        id="Proceedings_In_Established_Conference_Series",
        desc="Provides URL evidence that proceedings are published in an established, reputable conference series (as required).",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The proceedings are published in an established reputable conference series (e.g., {_safe(pq.series_name)}).",
        node=series_leaf,
        sources=series_sources,
        additional_instruction="Look for series branding such as IEEE CPS, ACM ICPS, LNCS/Springer, etc."
    )

    # Venue_In_US_Or_Europe
    venue_sources = _merge_sources(_urls(ll.venue_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Venue_In_US_Or_Europe_Sources_Provided",
        "Source URLs provided for venue location.",
        venue_sources
    )
    venue_leaf = evaluator.add_leaf(
        id="Venue_In_US_Or_Europe",
        desc="Provides the venue location (city/country) and a URL showing it is in the United States or Europe.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {_safe(ll.venue_location)} is located in the United States or Europe.",
        node=venue_leaf,
        sources=venue_sources,
        additional_instruction="Confirm location is in US or a European country; consider broader Europe definition (including UK)."
    )

    # Wheelchair_Accessible_Entrances_And_Exits
    access_sources = _merge_sources(_urls(ll.accessibility_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Wheelchair_Accessible_Entrances_And_Exits_Sources_Provided",
        "Source URLs provided for wheelchair-accessible entrances/exits.",
        access_sources
    )
    access_leaf = evaluator.add_leaf(
        id="Wheelchair_Accessible_Entrances_And_Exits",
        desc="Provides explicit venue/accessibility information (with URL evidence) confirming wheelchair-accessible entrances/exits (not assumed/\"standard\").",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue explicitly confirms wheelchair-accessible entrances and exits.",
        node=access_leaf,
        sources=access_sources,
        additional_instruction="Look for explicit accessibility statements; general marketing claims without specifics do not suffice."
    )

    # Negotiated_Hotel_Rates_Available
    hotel_sources = _merge_sources(_urls(ll.hotel_rates_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Negotiated_Hotel_Rates_Available_Sources_Provided",
        "Source URLs provided for negotiated hotel rates.",
        hotel_sources
    )
    hotel_leaf = evaluator.add_leaf(
        id="Negotiated_Hotel_Rates_Available",
        desc="Provides URL evidence of hotel accommodations at negotiated conference rates (e.g., room block/booking link/rate info).",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Hotel accommodations at negotiated conference rates (room block or special rates) are available.",
        node=hotel_leaf,
        sources=hotel_sources,
        additional_instruction="Look for booking links or text indicating negotiated rates for attendees."
    )

    # Student_Volunteer_Program_With_Benefits
    sv_sources = _merge_sources(_urls(ss.student_volunteer_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Student_Volunteer_Program_With_Benefits_Sources_Provided",
        "Source URLs provided for student volunteer program and benefits.",
        sv_sources
    )
    sv_leaf = evaluator.add_leaf(
        id="Student_Volunteer_Program_With_Benefits",
        desc="Provides URL evidence that a student volunteer program is offered and that benefits are defined.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="A student volunteer program is offered and its benefits are defined.",
        node=sv_leaf,
        sources=sv_sources,
        additional_instruction="Benefits may include fee waivers, access, meals, etc. Evidence must be explicit."
    )

    # Visa_Support_Letters_Available
    visa_sources = _merge_sources(_urls(ss.visa_support_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Visa_Support_Letters_Available_Sources_Provided",
        "Source URLs provided for visa support letters availability.",
        visa_sources
    )
    visa_leaf = evaluator.add_leaf(
        id="Visa_Support_Letters_Available",
        desc="Provides URL evidence that visa support letters are available for international attendees (not assumed).",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Visa support letters are available for international attendees.",
        node=visa_leaf,
        sources=visa_sources,
        additional_instruction="Accept pages describing visa invitation letters or official support process."
    )

    # Dietary_Restrictions_Accommodated
    diet_sources = _merge_sources(_urls(ss.dietary_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Dietary_Restrictions_Accommodated_Sources_Provided",
        "Source URLs provided for dietary accommodations.",
        diet_sources
    )
    diet_leaf = evaluator.add_leaf(
        id="Dietary_Restrictions_Accommodated",
        desc="Provides URL evidence that dietary restrictions are accommodated for conference meals/receptions.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Dietary restrictions are accommodated for conference meals and receptions.",
        node=diet_leaf,
        sources=diet_sources,
        additional_instruction="Look for explicit accommodation statements (vegetarian, vegan, halal, kosher, allergies, etc.)."
    )

    # Oral_And_Poster_Options_Available
    op_sources = _merge_sources(_urls(pr.oral_poster_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Oral_And_Poster_Options_Available_Sources_Provided",
        "Source URLs provided for oral and poster options.",
        op_sources
    )
    op_leaf = evaluator.add_leaf(
        id="Oral_And_Poster_Options_Available",
        desc="Provides URL evidence that both oral and poster presentation options exist.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Both oral and poster presentation options are available.",
        node=op_leaf,
        sources=op_sources,
        additional_instruction="Evidence can be in program formats, author guidelines, or presentation instructions."
    )

    # Poster_Size_Max_48_Inches
    poster_sources = _merge_sources(_urls(pr.poster_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Poster_Size_Max_48_Inches_Sources_Provided",
        "Source URLs provided for poster size maximum.",
        poster_sources
    )
    poster_leaf = evaluator.add_leaf(
        id="Poster_Size_Max_48_Inches",
        desc="Provides poster size rules (with URL evidence) showing no dimension exceeds 48 inches.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The poster size requirement does not exceed 48 inches in any dimension; the stated maximum is {_safe(pr.poster_max_size_in)}.",
        node=poster_leaf,
        sources=poster_sources,
        additional_instruction="If sizes are in centimeters or paper sizes, convert or interpret; any single dimension >48 inches fails the requirement."
    )

    # Oral_Presentation_Time_Allocations_Defined
    oral_time_sources = _merge_sources(_urls(pr.oral_time_alloc_urls), fallback=ident.official_url)
    _add_sources_gate(
        evaluator, cc_node,
        "Oral_Presentation_Time_Allocations_Defined_Sources_Provided",
        "Source URLs provided for oral presentation time allocations.",
        oral_time_sources
    )
    oral_time_leaf = evaluator.add_leaf(
        id="Oral_Presentation_Time_Allocations_Defined",
        desc="Provides URL evidence that oral presentation time allocations are clearly defined.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim="Oral presentations have clearly defined time allocations.",
        node=oral_time_leaf,
        sources=oral_time_sources,
        additional_instruction="Accept explicit slot lengths in program or presenter guidelines."
    )


# =========================
# Main Evaluation Entry
# =========================
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
        default_model=model
    )

    # Extract structured info from the answer
    ext = await evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceExtraction,
        extraction_name="conference_extraction"
    )

    # Build top-level "Conference_Selection" aggregator as per rubric
    conf_sel_node = evaluator.add_sequential(
        id="Conference_Selection",
        desc="Answer identifies exactly one specific major AI/ML conference in 2026 that satisfies all listed constraints, and provides verifiable evidence with URLs.",
        parent=root,
        critical=True
    )

    # Conference identification checks
    await build_conference_identification(evaluator, conf_sel_node, ext)

    # Constraint compliance checks (only evaluated if identification passes due to sequential gating)
    await build_constraint_checks(evaluator, conf_sel_node, ext)

    return evaluator.get_summary()