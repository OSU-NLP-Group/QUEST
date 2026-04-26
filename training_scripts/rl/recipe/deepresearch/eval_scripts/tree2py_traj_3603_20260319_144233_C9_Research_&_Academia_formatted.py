import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# =============================================================================
# Task constants
# =============================================================================
TASK_ID = "grad_student_march_2026_tasks"
TASK_DESCRIPTION = (
    "A PhD student in lunar science at a U.S. university is conducting research on the Chang'e-6 farside samples "
    "from the Apollo Basin. In March 2026, the student needs to accomplish four parallel academic tasks:\n\n"
    "1. Journal Manuscript Submission: Identify a suitable peer-reviewed astronomy journal for submitting a research "
    "manuscript on Chang'e-6 basalt samples. The journal must have an impact factor of at least 4.0, publish lunar/"
    "planetary science research, and have a peer review timeline that accommodates the student's schedule. The student "
    "also needs to understand the journal's open access policy and whether APC waivers are available.\n\n"
    "2. NASA Space Grant Fellowship: The student must apply for a NASA Space Grant graduate fellowship that offers at "
    "least $10,000 for the academic year. The fellowship application deadline must be before March 16, 2026 (when the "
    "student will attend LPSC 2026), and the fellowship must require U.S. citizenship and support NASA-relevant research "
    "in STEM fields.\n\n"
    "3. LPSC 2026 Conference Session: The student plans to attend the 57th Lunar and Planetary Science Conference (LPSC "
    "2026) in The Woodlands, Texas (March 16-20, 2026). Identify the most relevant conference session for presenting "
    "research on Chang'e-6 lunar sample return and analysis. The session should focus on lunar science and be suitable "
    "for research presentations (oral or special session format).\n\n"
    "4. Total Lunar Eclipse Observation: The student wants to observe the total lunar eclipse occurring on March 3, 2026. "
    "Identify a location where the complete totality phase (approximately 58 minutes duration) is visible, provide the "
    "local times for totality, and confirm that the eclipse observation does not conflict with the fellowship application "
    "deadline or conference attendance.\n\n"
    "For each of the four items, provide: the specific name/title and identifying details, all relevant verification "
    "information (dates, times, amounts, durations, etc.), and a reference URL supporting each piece of information."
)

LPSC_2026_START = "2026-03-16"
LPSC_2026_END = "2026-03-20"
ECLIPSE_DATE = "2026-03-03"


# =============================================================================
# Data models for extraction
# =============================================================================
class JournalExtraction(BaseModel):
    name: Optional[str] = None
    identifying_detail: Optional[str] = None  # e.g., publisher, ISSN
    journal_urls: List[str] = Field(default_factory=list)  # official pages for the journal

    peer_review_statement: Optional[str] = None
    peer_review_urls: List[str] = Field(default_factory=list)

    astronomy_scope_statement: Optional[str] = None
    astronomy_scope_urls: List[str] = Field(default_factory=list)

    lunar_planetary_scope_statement: Optional[str] = None
    lunar_planetary_scope_urls: List[str] = Field(default_factory=list)

    impact_factor_value: Optional[str] = None  # keep as free text; can be "5.2 (2023)" etc.
    impact_factor_urls: List[str] = Field(default_factory=list)

    peer_review_timeline_info: Optional[str] = None  # e.g., "median 6 weeks", "first decision ~30 days"
    peer_review_timeline_urls: List[str] = Field(default_factory=list)

    open_access_policy: Optional[str] = None
    open_access_urls: List[str] = Field(default_factory=list)

    apc_waiver_info: Optional[str] = None
    apc_waiver_urls: List[str] = Field(default_factory=list)


class FellowshipExtraction(BaseModel):
    fellowship_name: Optional[str] = None
    identifying_detail: Optional[str] = None  # e.g., state consortium or host org
    program_urls: List[str] = Field(default_factory=list)

    nasa_space_grant_statement: Optional[str] = None
    nasa_space_grant_urls: List[str] = Field(default_factory=list)

    award_amount_text: Optional[str] = None
    award_amount_urls: List[str] = Field(default_factory=list)

    deadline_date_text: Optional[str] = None  # keep in free text, e.g., "March 10, 2026" or "2026-03-10"
    deadline_urls: List[str] = Field(default_factory=list)

    citizenship_requirement_text: Optional[str] = None
    citizenship_urls: List[str] = Field(default_factory=list)

    supports_nasa_research_statement: Optional[str] = None
    supports_research_urls: List[str] = Field(default_factory=list)


class LPSCSessionExtraction(BaseModel):
    session_title: Optional[str] = None
    session_identifier: Optional[str] = None  # e.g., session number or code
    program_urls: List[str] = Field(default_factory=list)

    official_session_evidence: Optional[str] = None
    official_session_urls: List[str] = Field(default_factory=list)

    lunar_focus_statement: Optional[str] = None
    lunar_focus_urls: List[str] = Field(default_factory=list)

    sample_return_relevance_statement: Optional[str] = None
    sample_return_urls: List[str] = Field(default_factory=list)

    session_format_statement: Optional[str] = None  # e.g., "Oral", "Special Session"
    session_format_urls: List[str] = Field(default_factory=list)


class EclipseExtraction(BaseModel):
    location_name: Optional[str] = None
    location_identifying_details: Optional[str] = None  # e.g., country/region/coordinates
    location_urls: List[str] = Field(default_factory=list)

    totality_visible_statement: Optional[str] = None
    totality_visible_urls: List[str] = Field(default_factory=list)

    totality_start_local: Optional[str] = None
    totality_end_local: Optional[str] = None
    totality_times_urls: List[str] = Field(default_factory=list)

    totality_duration_statement: Optional[str] = None  # e.g., "58 minutes", "~58 min", "00:58"
    totality_duration_urls: List[str] = Field(default_factory=list)


# =============================================================================
# Extraction prompts
# =============================================================================
def prompt_extract_journal() -> str:
    return """
Extract one suitable peer-reviewed astronomy journal for a manuscript on Chang'e-6 basalt samples. From the answer, extract ONLY what is explicitly stated and their cited URLs.

Return a JSON with fields:
- name: the specific journal name
- identifying_detail: at least one identifying detail such as publisher, ISSN, or the journal's official page name/identifier
- journal_urls: array of official journal URLs (homepage, aims/scope, about, etc.)

- peer_review_statement: a sentence/phrase explicitly stating the journal is peer-reviewed or has peer review
- peer_review_urls: array of URLs that support the peer review claim

- astronomy_scope_statement: a sentence/phrase showing it is an astronomy/astrophysics journal
- astronomy_scope_urls: array of URLs supporting it is an astronomy/astrophysics journal

- lunar_planetary_scope_statement: a sentence/phrase showing it publishes lunar and/or planetary science
- lunar_planetary_scope_urls: array of URLs supporting lunar/planetary science scope

- impact_factor_value: the impact factor (as text, e.g., "5.2 (2023)")
- impact_factor_urls: array of URLs that state the impact factor

- peer_review_timeline_info: typical peer review timeline text (e.g., "median 6 weeks", "time to first decision ~30 days")
- peer_review_timeline_urls: array of URLs supporting the timeline

- open_access_policy: summary text of the journal's OA policy
- open_access_urls: array of URLs supporting OA policy

- apc_waiver_info: text describing APC waivers/discounts (if any)
- apc_waiver_urls: array of URLs supporting APC waiver information

Rules:
- Extract only URLs explicitly present. If a URL is missing, leave the array empty.
- Do not invent numbers or URLs. If not present, return null or empty array accordingly.
"""


def prompt_extract_fellowship() -> str:
    return """
Extract one NASA Space Grant graduate fellowship program meeting the task requirements from the answer with supporting URLs.

Return a JSON with fields:
- fellowship_name: the specific fellowship or program title
- identifying_detail: state consortium or host organization or similar identifier
- program_urls: array of official program URLs

- nasa_space_grant_statement: phrase indicating it is a NASA Space Grant program (e.g., part of a state consortium)
- nasa_space_grant_urls: array of URLs supporting NASA Space Grant affiliation

- award_amount_text: the stated award amount (text). It must be at least $10,000 for the academic year.
- award_amount_urls: array of URLs supporting the amount

- deadline_date_text: the application deadline date as text (e.g., "March 10, 2026" or "2026-03-10")
- deadline_urls: array of URLs supporting the deadline

- citizenship_requirement_text: text stating U.S. citizenship is required
- citizenship_urls: array of URLs supporting the citizenship requirement

- supports_nasa_research_statement: text stating the fellowship supports NASA-relevant STEM research
- supports_research_urls: array of URLs supporting this statement

Rules:
- Extract only what is explicitly in the answer and the associated URLs.
- Do not guess or create fields; leave missing items as null or empty arrays.
"""


def prompt_extract_lpsc_session() -> str:
    return """
Extract one LPSC 2026 session suitable for Chang'e-6 lunar sample return/analysis research with supporting URLs.

Return a JSON with fields:
- session_title: specific session title
- session_identifier: session number/code or other identifier from the program
- program_urls: array of URLs to the official LPSC 2026 technical program/listing

- official_session_evidence: phrase confirming it is an official LPSC 2026 session
- official_session_urls: array of URLs supporting official status (e.g., technical program page)

- lunar_focus_statement: phrase confirming the session focuses on lunar science
- lunar_focus_urls: array of URLs supporting lunar focus

- sample_return_relevance_statement: phrase confirming relevance to lunar sample return/analysis
- sample_return_urls: array of URLs supporting relevance to sample return/analysis

- session_format_statement: phrase describing the format (e.g., "Oral Session", "Special Session")
- session_format_urls: array of URLs supporting session format and suitability for research presentations

Rules:
- Use URLs explicitly cited in the answer.
- If not available, leave as empty arrays or null.
"""


def prompt_extract_eclipse() -> str:
    return """
Extract a location and details for observing the March 3, 2026 total lunar eclipse with supporting URLs.

Return a JSON with fields:
- location_name: specific observation location name (city/observatory/site)
- location_identifying_details: country/region and/or coordinates or other identifying info
- location_urls: array of URLs supporting the identification of the location

- totality_visible_statement: phrase confirming the complete totality is visible from this location
- totality_visible_urls: array of URLs supporting this visibility

- totality_start_local: local time for totality start at the chosen location
- totality_end_local: local time for totality end at the chosen location
- totality_times_urls: array of URLs supporting local totality times

- totality_duration_statement: text indicating totality duration (e.g., about 58 minutes)
- totality_duration_urls: array of URLs supporting duration

Rules:
- Extract only information explicitly present in the answer and its URLs.
- Keep times and durations as presented (strings). Do not infer or convert.
"""


# =============================================================================
# Helper utilities
# =============================================================================
def _urls_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    filtered = [u for u in urls if isinstance(u, str) and u.strip()]
    return filtered if filtered else None


def _add_ins_with_url_requirement(base: str, urls: Optional[List[str]]) -> str:
    if urls and len(urls) > 0:
        return base
    # If no URL present, instruct judge to mark as not supported
    return base + "\nImportant: No supporting URL was provided for this verification. You must judge the claim as NOT SUPPORTED."


# =============================================================================
# Verification builders per item
# =============================================================================
async def verify_item_1_journal(evaluator: Evaluator, parent) -> None:
    item_node = evaluator.add_parallel(
        id="Item_1_Journal_Manuscript_Submission",
        desc="Identify a suitable peer-reviewed astronomy journal meeting impact factor, scope, timeline, and OA/APC-waiver requirements, with supporting URLs.",
        parent=parent,
        critical=False
    )

    journal: JournalExtraction = await evaluator.extract(
        prompt=prompt_extract_journal(),
        template_class=JournalExtraction,
        extraction_name="journal_extraction"
    )

    # 1) Journal name + identifying detail + URL
    leaf_1 = evaluator.add_leaf(
        id="Journal_Name_And_Identifying_Details_With_URL",
        desc="Provides the journal’s specific name and at least one identifying detail and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_1 = _urls_or_none(journal.journal_urls)
    claim_1 = f"The journal is named '{journal.name}' and has identifying detail '{journal.identifying_detail}'. A supporting URL is provided."
    await evaluator.verify(
        claim=claim_1,
        node=leaf_1,
        sources=urls_1,
        additional_instruction=_add_ins_with_url_requirement(
            "Confirm the journal name and identifying detail on the provided official journal page(s).",
            urls_1
        )
    )

    # 2) Peer-reviewed + URL
    leaf_2 = evaluator.add_leaf(
        id="Journal_Is_Peer_Reviewed_With_URL",
        desc="Confirms the journal is peer-reviewed and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_2 = _urls_or_none(journal.peer_review_urls)
    claim_2 = "The journal is peer-reviewed (has a formal peer review process)."
    await evaluator.verify(
        claim=claim_2,
        node=leaf_2,
        sources=urls_2,
        additional_instruction=_add_ins_with_url_requirement(
            "Look for explicit statements like 'peer-reviewed', 'refereed', or description of peer review workflow.",
            urls_2
        )
    )

    # 3) Astronomy journal + URL
    leaf_3 = evaluator.add_leaf(
        id="Journal_Is_Astronomy_Journal_With_URL",
        desc="Confirms the journal is an astronomy/astrophysics journal and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_3 = _urls_or_none(journal.astronomy_scope_urls)
    claim_3 = "This is an astronomy or astrophysics journal."
    await evaluator.verify(
        claim=claim_3,
        node=leaf_3,
        sources=urls_3,
        additional_instruction=_add_ins_with_url_requirement(
            "Use Aims & Scope or About pages to confirm the astronomy/astrophysics domain.",
            urls_3
        )
    )

    # 4) Publishes lunar/planetary science + URL
    leaf_4 = evaluator.add_leaf(
        id="Journal_Publishes_Lunar_Or_Planetary_Science_With_URL",
        desc="Confirms the journal publishes lunar and/or planetary science research and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_4 = _urls_or_none(journal.lunar_planetary_scope_urls)
    claim_4 = "The journal publishes lunar and/or planetary science research."
    await evaluator.verify(
        claim=claim_4,
        node=leaf_4,
        sources=urls_4,
        additional_instruction=_add_ins_with_url_requirement(
            "Check scope or example articles showing lunar/planetary science content.",
            urls_4
        )
    )

    # 5) Impact factor >= 4 + URL
    leaf_5 = evaluator.add_leaf(
        id="Impact_Factor_At_Least_4_With_URL",
        desc="Provides an impact factor value and confirms it is ≥ 4.0 and cites a supporting URL for the impact factor.",
        parent=item_node,
        critical=True
    )
    urls_5 = _urls_or_none(journal.impact_factor_urls)
    claim_5 = f"The journal has an impact factor of {journal.impact_factor_value}, which is at least 4.0."
    await evaluator.verify(
        claim=claim_5,
        node=leaf_5,
        sources=urls_5,
        additional_instruction=_add_ins_with_url_requirement(
            "Verify the impact factor value and ensure it is ≥ 4.0. Accept reasonable year variations if clearly stated.",
            urls_5
        )
    )

    # 6) Peer review timeline accommodates schedule + URL
    leaf_6 = evaluator.add_leaf(
        id="Peer_Review_Timeline_Accommodates_Schedule_With_URL",
        desc="Provides peer-review timeline information and indicates it accommodates the student's schedule, and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_6 = _urls_or_none(journal.peer_review_timeline_urls)
    claim_6 = f"The journal's typical peer review timeline is: {journal.peer_review_timeline_info}. This timeline can accommodate a March 2026 submission schedule."
    await evaluator.verify(
        claim=claim_6,
        node=leaf_6,
        sources=urls_6,
        additional_instruction=_add_ins_with_url_requirement(
            "Focus on verifying the stated review timeline from the journal's official pages or credible sources.",
            urls_6
        )
    )

    # 7) Open access policy + URL
    leaf_7 = evaluator.add_leaf(
        id="Open_Access_Policy_Stated_With_URL",
        desc="States the journal’s open access policy and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_7 = _urls_or_none(journal.open_access_urls)
    claim_7 = f"The journal's open access policy states: {journal.open_access_policy}"
    await evaluator.verify(
        claim=claim_7,
        node=leaf_7,
        sources=urls_7,
        additional_instruction=_add_ins_with_url_requirement(
            "Confirm OA policy details from the journal or publisher's official pages.",
            urls_7
        )
    )

    # 8) APC waiver availability + URL
    leaf_8 = evaluator.add_leaf(
        id="APC_Waiver_Availability_Stated_With_URL",
        desc="States whether APC waivers (or equivalent fee relief) are available and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_8 = _urls_or_none(journal.apc_waiver_urls)
    claim_8 = f"APC waivers or discounts availability: {journal.apc_waiver_info}"
    await evaluator.verify(
        claim=claim_8,
        node=leaf_8,
        sources=urls_8,
        additional_instruction=_add_ins_with_url_requirement(
            "Verify any waiver/discount/low-income country policy on official pages.",
            urls_8
        )
    )


async def verify_item_2_fellowship(evaluator: Evaluator, parent) -> None:
    item_node = evaluator.add_parallel(
        id="Item_2_NASA_Space_Grant_Fellowship",
        desc="Identify a NASA Space Grant graduate fellowship meeting amount, deadline, citizenship, and research scope requirements, with supporting URLs.",
        parent=parent,
        critical=False
    )

    fel: FellowshipExtraction = await evaluator.extract(
        prompt=prompt_extract_fellowship(),
        template_class=FellowshipExtraction,
        extraction_name="fellowship_extraction"
    )

    # 1) Fellowship name + identifying details + URL
    leaf_1 = evaluator.add_leaf(
        id="Fellowship_Name_And_Identifying_Details_With_URL",
        desc="Provides the specific fellowship/program name and identifying details and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_1 = _urls_or_none(fel.program_urls)
    claim_1 = f"The fellowship is '{fel.fellowship_name}' identified by '{fel.identifying_detail}'. A supporting URL is provided."
    await evaluator.verify(
        claim=claim_1,
        node=leaf_1,
        sources=urls_1,
        additional_instruction=_add_ins_with_url_requirement(
            "Confirm the program name and identifying consortium/host details on the official program page.",
            urls_1
        )
    )

    # 2) Is NASA Space Grant + URL
    leaf_2 = evaluator.add_leaf(
        id="Fellowship_Is_NASA_Space_Grant_With_URL",
        desc="Confirms the fellowship is part of NASA Space Grant and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_2 = _urls_or_none(fel.nasa_space_grant_urls)
    claim_2 = "This fellowship is part of the NASA Space Grant program (e.g., state consortium)."
    await evaluator.verify(
        claim=claim_2,
        node=leaf_2,
        sources=urls_2,
        additional_instruction=_add_ins_with_url_requirement(
            "Look for explicit connection to NASA Space Grant, state consortium branding, or NASA listing.",
            urls_2
        )
    )

    # 3) Award amount >= $10,000 + URL
    leaf_3 = evaluator.add_leaf(
        id="Award_Amount_At_Least_10000_With_URL",
        desc="Provides the award amount and confirms it is ≥ $10,000 for the academic year and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_3 = _urls_or_none(fel.award_amount_urls)
    claim_3 = f"The fellowship award amount is {fel.award_amount_text}, which is at least $10,000 for the academic year."
    await evaluator.verify(
        claim=claim_3,
        node=leaf_3,
        sources=urls_3,
        additional_instruction=_add_ins_with_url_requirement(
            "Verify the total academic-year award (not monthly) meets or exceeds $10,000.",
            urls_3
        )
    )

    # 4) Deadline before March 16, 2026 + URL
    leaf_4 = evaluator.add_leaf(
        id="Deadline_Before_March_16_2026_With_URL",
        desc="Provides the application deadline date and confirms it is before March 16, 2026 and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_4 = _urls_or_none(fel.deadline_urls)
    claim_4 = f"The application deadline is {fel.deadline_date_text}, which is before March 16, 2026."
    await evaluator.verify(
        claim=claim_4,
        node=leaf_4,
        sources=urls_4,
        additional_instruction=_add_ins_with_url_requirement(
            "Check the posted deadline date and judge if it occurs strictly before 2026-03-16.",
            urls_4
        )
    )

    # 5) Requires U.S. citizenship + URL
    leaf_5 = evaluator.add_leaf(
        id="Requires_US_Citizenship_With_URL",
        desc="States that U.S. citizenship is required for eligibility and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_5 = _urls_or_none(fel.citizenship_urls)
    claim_5 = "U.S. citizenship is required for fellowship eligibility."
    await evaluator.verify(
        claim=claim_5,
        node=leaf_5,
        sources=urls_5,
        additional_instruction=_add_ins_with_url_requirement(
            "Confirm explicit eligibility requirement for U.S. citizens.",
            urls_5
        )
    )

    # 6) Supports NASA-relevant STEM research + URL
    leaf_6 = evaluator.add_leaf(
        id="Supports_NASA_Relevant_STEM_Research_With_URL",
        desc="States that the fellowship supports NASA-relevant research in STEM fields and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_6 = _urls_or_none(fel.supports_research_urls)
    claim_6 = "The fellowship supports NASA-relevant research in STEM fields."
    await evaluator.verify(
        claim=claim_6,
        node=leaf_6,
        sources=urls_6,
        additional_instruction=_add_ins_with_url_requirement(
            "Look for program goals or eligibility text referencing STEM and NASA relevance.",
            urls_6
        )
    )


async def verify_item_3_lpsc_session(evaluator: Evaluator, parent) -> None:
    item_node = evaluator.add_parallel(
        id="Item_3_LPSC_2026_Session",
        desc="Identify the most relevant LPSC 2026 session for Chang'e-6 sample return/analysis.",
        parent=parent,
        critical=False
    )

    sess: LPSCSessionExtraction = await evaluator.extract(
        prompt=prompt_extract_lpsc_session(),
        template_class=LPSCSessionExtraction,
        extraction_name="lpsc_session_extraction"
    )

    # 1) Session title and identifying details + URL
    leaf_1 = evaluator.add_leaf(
        id="Session_Title_And_Identifying_Details_With_URL",
        desc="Provides the session’s specific title and identifying details and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_1 = _urls_or_none(sess.program_urls)
    claim_1 = f"The LPSC 2026 session titled '{sess.session_title}' has identifier '{sess.session_identifier}', as listed in the official program."
    await evaluator.verify(
        claim=claim_1,
        node=leaf_1,
        sources=urls_1,
        additional_instruction=_add_ins_with_url_requirement(
            "Verify that the session title and identifier appear on the official LPSC 2026 program/listing.",
            urls_1
        )
    )

    # 2) Official LPSC 2026 session + URL
    leaf_2 = evaluator.add_leaf(
        id="Session_Is_Official_LPSC_2026_Session_With_URL",
        desc="Provides evidence the session is an official LPSC 2026 session and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_2 = _urls_or_none(sess.official_session_urls if sess.official_session_urls else sess.program_urls)
    claim_2 = "This is an official session of the 57th Lunar and Planetary Science Conference (LPSC 2026)."
    await evaluator.verify(
        claim=claim_2,
        node=leaf_2,
        sources=urls_2,
        additional_instruction=_add_ins_with_url_requirement(
            "Confirm that the session is part of the official LPSC 2026 technical program.",
            urls_2
        )
    )

    # 3) Session focuses on lunar science + URL
    leaf_3 = evaluator.add_leaf(
        id="Session_Focuses_On_Lunar_Science_With_URL",
        desc="Confirms the session focus is lunar science and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_3 = _urls_or_none(sess.lunar_focus_urls if sess.lunar_focus_urls else sess.program_urls)
    claim_3 = "The session focuses on lunar science."
    await evaluator.verify(
        claim=claim_3,
        node=leaf_3,
        sources=urls_3,
        additional_instruction=_add_ins_with_url_requirement(
            "Confirm that the session description or title clearly targets lunar science.",
            urls_3
        )
    )

    # 4) Relevant to lunar sample return/analysis + URL
    leaf_4 = evaluator.add_leaf(
        id="Session_Relevant_To_Sample_Return_Analysis_With_URL",
        desc="Confirms the session is suitable for lunar sample return and analysis research and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_4 = _urls_or_none(sess.sample_return_urls if sess.sample_return_urls else sess.program_urls)
    claim_4 = "The session is suitable for presenting lunar sample return and analysis research (e.g., Chang'e-6)."
    await evaluator.verify(
        claim=claim_4,
        node=leaf_4,
        sources=urls_4,
        additional_instruction=_add_ins_with_url_requirement(
            "Look for references to sample science, returned samples, or related analysis in title/description.",
            urls_4
        )
    )

    # 5) Session format suitable for research presentations + URL
    leaf_5 = evaluator.add_leaf(
        id="Session_Format_Suitable_For_Presentation_With_URL",
        desc="States the session format and confirms it is appropriate for research presentations (oral or special session) and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_5 = _urls_or_none(sess.session_format_urls if sess.session_format_urls else sess.program_urls)
    claim_5 = f"The session format is '{sess.session_format_statement}', which is appropriate for research presentations."
    await evaluator.verify(
        claim=claim_5,
        node=leaf_5,
        sources=urls_5,
        additional_instruction=_add_ins_with_url_requirement(
            "Confirm the session format (oral/special session) from the official program page.",
            urls_5
        )
    )


async def verify_item_4_eclipse(evaluator: Evaluator, parent, fellowship_deadline_text: Optional[str]) -> None:
    item_node = evaluator.add_parallel(
        id="Item_4_Total_Lunar_Eclipse_Observation",
        desc="Plan observation of the March 3, 2026 total lunar eclipse with location, times, ~58-min totality, and no conflicts.",
        parent=parent,
        critical=False
    )

    ecl: EclipseExtraction = await evaluator.extract(
        prompt=prompt_extract_eclipse(),
        template_class=EclipseExtraction,
        extraction_name="eclipse_extraction"
    )

    # 1) Observation location name and identifying details + URL
    leaf_1 = evaluator.add_leaf(
        id="Observation_Location_Name_And_Identifying_Details_With_URL",
        desc="Provides a specific observation location name and identifying details and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_1 = _urls_or_none(ecl.location_urls)
    claim_1 = f"The observation location is '{ecl.location_name}' identified by '{ecl.location_identifying_details}'."
    await evaluator.verify(
        claim=claim_1,
        node=leaf_1,
        sources=urls_1,
        additional_instruction=_add_ins_with_url_requirement(
            "Confirm the location name and identifying details (region/country/coordinates) on the provided page.",
            urls_1
        )
    )

    # 2) Complete totality visible at location + URL
    leaf_2 = evaluator.add_leaf(
        id="Complete_Totality_Visible_At_Location_With_URL",
        desc="Confirms that the complete totality phase is visible from the chosen location and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_2 = _urls_or_none(ecl.totality_visible_urls if ecl.totality_visible_urls else ecl.totality_times_urls)
    claim_2 = f"At {ecl.location_name}, the complete totality phase of the March 3, 2026 total lunar eclipse is visible."
    await evaluator.verify(
        claim=claim_2,
        node=leaf_2,
        sources=urls_2,
        additional_instruction=_add_ins_with_url_requirement(
            "Use eclipse maps/tables to confirm the location lies within the totality visibility footprint.",
            urls_2
        )
    )

    # 3) Local totality times provided + URL
    leaf_3 = evaluator.add_leaf(
        id="Local_Totality_Times_Provided_With_URL",
        desc="Provides local start and end times for totality at the chosen location and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_3 = _urls_or_none(ecl.totality_times_urls)
    claim_3 = f"At {ecl.location_name}, the local totality times on 2026-03-03 are {ecl.totality_start_local} to {ecl.totality_end_local}."
    await evaluator.verify(
        claim=claim_3,
        node=leaf_3,
        sources=urls_3,
        additional_instruction=_add_ins_with_url_requirement(
            "Confirm the local start and end for totality at this location on March 3, 2026.",
            urls_3
        )
    )

    # 4) Totality duration approximately 58 minutes + URL
    leaf_4 = evaluator.add_leaf(
        id="Totality_Duration_Approximately_58_Minutes_With_URL",
        desc="States totality duration and confirms it is approximately 58 minutes and cites a supporting URL.",
        parent=item_node,
        critical=True
    )
    urls_4 = _urls_or_none(ecl.totality_duration_urls if ecl.totality_duration_urls else ecl.totality_times_urls)
    claim_4 = "The totality duration of the March 3, 2026 lunar eclipse is approximately 58 minutes."
    await evaluator.verify(
        claim=claim_4,
        node=leaf_4,
        sources=urls_4,
        additional_instruction=_add_ins_with_url_requirement(
            "Allow small rounding differences; ~58 minutes is acceptable if within a few minutes.",
            urls_4
        )
    )

    # 5) No conflict with LPSC dates (simple logical verification; no URL required)
    leaf_5 = evaluator.add_leaf(
        id="No_Conflict_With_LPSC_Dates",
        desc="Explicitly confirms the eclipse observation does not conflict with LPSC attendance dates (March 16–20, 2026).",
        parent=item_node,
        critical=True
    )
    claim_5 = (
        f"The total lunar eclipse occurs on {ECLIPSE_DATE}, and LPSC 2026 occurs on {LPSC_2026_START}–{LPSC_2026_END}; "
        "there is no scheduling conflict because they are on different dates."
    )
    await evaluator.verify(
        claim=claim_5,
        node=leaf_5,
        sources=None,
        additional_instruction="This is a straightforward date comparison; confirm the dates are distinct with no overlap."
    )

    # 6) No conflict with fellowship deadline chosen (simple logical verification; no URL required)
    leaf_6 = evaluator.add_leaf(
        id="No_Conflict_With_Fellowship_Deadline_Chosen",
        desc="Explicitly confirms the eclipse observation does not conflict with the fellowship application deadline date chosen in Item 2.",
        parent=item_node,
        critical=True
    )
    claim_6 = (
        f"The fellowship application deadline is {fellowship_deadline_text}; the eclipse is on {ECLIPSE_DATE}. "
        "These do not conflict if the deadline is not on the same calendar date as the eclipse at the observation location."
    )
    await evaluator.verify(
        claim=claim_6,
        node=leaf_6,
        sources=None,
        additional_instruction="Judge this by simple date equality: if the deadline date equals 2026-03-03 (local), it's a conflict; otherwise, it is not."
    )


# =============================================================================
# Main evaluation entry point
# =============================================================================
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

    # Build top-level node per rubric
    top = evaluator.add_parallel(
        id="Graduate_Student_March_2026_Tasks",
        desc="Complete four academic tasks for March 2026: journal selection, Space Grant fellowship selection, LPSC 2026 session selection, and eclipse observation planning.",
        parent=root,
        critical=False
    )

    # We need fellowship deadline text extracted for conflict check in Item 4
    # Extract fellowship first to pass deadline into eclipse checks
    fellowship_extraction_task = evaluator.extract(
        prompt=prompt_extract_fellowship(),
        template_class=FellowshipExtraction,
        extraction_name="fellowship_extraction_pre"
    )

    # Extract other items in parallel while fellowship is extracted
    journal_and_session_and_eclipse = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_journal(),
            template_class=JournalExtraction,
            extraction_name="journal_extraction_pre"
        ),
        evaluator.extract(
            prompt=prompt_extract_lpsc_session(),
            template_class=LPSCSessionExtraction,
            extraction_name="lpsc_session_extraction_pre"
        ),
        evaluator.extract(
            prompt=prompt_extract_eclipse(),
            template_class=EclipseExtraction,
            extraction_name="eclipse_extraction_pre"
        ),
    )
    # Results above are recorded for transparency, but we will run dedicated per-item verification routines
    # to ensure node structure matches the rubric and verifications are properly linked.

    # Await fellowship extraction (ensure available for eclipse conflict text)
    fellowship_pre: FellowshipExtraction = await fellowship_extraction_task
    fellowship_deadline_text = fellowship_pre.deadline_date_text

    # Add ground truth style context info helpful for judge reasoning
    evaluator.add_custom_info(
        info={
            "LPSC_2026_dates": {"start": LPSC_2026_START, "end": LPSC_2026_END},
            "Eclipse_date": ECLIPSE_DATE
        },
        info_type="reference_dates",
        info_name="reference_dates_for_conflict_checks"
    )

    # Now run verification builders (each will internally re-extract the structured info used in their area,
    # which is fine for robustness and logging; cached LLM results may be reused by the framework).
    await asyncio.gather(
        verify_item_1_journal(evaluator, top),
        verify_item_2_fellowship(evaluator, top),
        verify_item_3_lpsc_session(evaluator, top),
        verify_item_4_eclipse(evaluator, top, fellowship_deadline_text),
    )

    return evaluator.get_summary()