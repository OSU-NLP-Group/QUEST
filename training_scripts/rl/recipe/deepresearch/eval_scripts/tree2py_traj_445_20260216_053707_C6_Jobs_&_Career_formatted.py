import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dean_positions_oh_pa_tn_2025_2026"
TASK_DESCRIPTION = (
    "Identify three current dean position openings (not Associate/Assistant/Acting) at universities in Ohio, Pennsylvania, or Tennessee, "
    "restricted to Colleges/Schools of Business, Education, or Arts & Sciences (or equivalent), requiring a terminal/doctoral degree, "
    "reporting directly to the Provost/SVPA for Academic Affairs/Chief Academic Officer, managed by a professional academic search firm, "
    "and posted between Nov 2025 and Feb 2026. For each, provide title, university, location, college/school, search firm, posted date, and official posting URL."
)

ALLOWED_STATES_FULL = {"Ohio": "OH", "Pennsylvania": "PA", "Tennessee": "TN"}
ALLOWED_STATE_ABBR = {"OH": "Ohio", "PA": "Pennsylvania", "TN": "Tennessee"}
ALLOWED_STATE_SET = set(ALLOWED_STATES_FULL.keys()) | set(ALLOWED_STATE_ABBR.keys())

# Time window (inclusive)
TIME_WINDOW_START = "2025-11-01"
TIME_WINDOW_END = "2026-02-28"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    position_title: Optional[str] = None
    university_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Accept abbreviations or full names
    college_school: Optional[str] = None  # e.g., College of Business, School of Education, College of Arts & Sciences
    search_firm: Optional[str] = None  # e.g., WittKieffer, Storbeck, Academic Search, Isaacson Miller
    posted_date: Optional[str] = None  # Prefer ISO or natural language; do not enforce strict format
    job_url: Optional[str] = None


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to the first three dean position entries provided in the answer. For each position, return:
    - position_title: The exact title as written in the answer (e.g., "Dean, College of Business").
    - university_name: The university or institution name.
    - city: The city of the institution.
    - state: The state (use full name like "Ohio" or two-letter abbreviation like "OH").
    - college_school: The specific college or school (e.g., "College of Business", "School of Education", "College of Arts & Sciences").
    - search_firm: The professional academic search firm or consultant managing the search (if named).
    - posted_date: The stated posting/announcement date as presented (any readable format).
    - job_url: The direct URL to the official job posting page (search firm page, university HR site, or widely used higher-ed job board).

    Rules:
    - Extract exactly what appears in the answer; do not infer or add missing information.
    - If any field is missing for a position, set it to null.
    - For job_url, only include a valid URL present in the answer. If a URL lacks "http" prefix, prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if s in ALLOWED_STATE_ABBR:
        return ALLOWED_STATE_ABBR[s]
    # Normalize casing for full names
    for full in ALLOWED_STATES_FULL:
        if s.lower() == full.lower():
            return full
    return s


def is_valid_http_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


# --------------------------------------------------------------------------- #
# Verification builder for a single position                                  #
# --------------------------------------------------------------------------- #
async def verify_single_position(
    evaluator: Evaluator,
    root_node,
    pos: PositionItem,
    position_index: int,
) -> None:
    """
    Build the verification subtree for a single dean position and perform all verifications.
    """
    pos_id = f"Position_{position_index + 1}"
    pos_node = evaluator.add_parallel(
        id=pos_id,
        desc=f"{['First','Second','Third'][position_index]} dean position meeting all requirements",
        parent=root_node,
        critical=False  # Each position contributes partial credit independently
    )

    # ----------------- Position Verification (Basic + Institution) -----------------
    pv_node = evaluator.add_parallel(
        id=f"P{position_index+1}_Position_Verification",
        desc="Verify the position details and university information",
        parent=pos_node,
        critical=True
    )

    # Basic Information
    basic_info_node = evaluator.add_parallel(
        id=f"P{position_index+1}_Basic_Information",
        desc="Basic job information verification",
        parent=pv_node,
        critical=True
    )

    # URL Reference
    if is_valid_http_url(pos.job_url):
        url_leaf = evaluator.add_leaf(
            id=f"P{position_index+1}_URL_Reference",
            desc=f"Valid URL to the position's official job posting",
            parent=basic_info_node,
            critical=True
        )
        await evaluator.verify(
            claim="This webpage is an official dean job posting hosted on a university site, a professional academic search firm's site, or a reputable higher-education job board.",
            node=url_leaf,
            sources=pos.job_url,
            additional_instruction="Treat sites like university domains, recognized search firms (e.g., WittKieffer, Storbeck, Academic Search, Isaacson Miller, RPA Inc., Greenwood/Asher), or reputable job boards (e.g., HigherEdJobs) as valid. The page should clearly be a job posting (not a news article, personal blog, or generic portal)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"P{position_index+1}_URL_Reference",
            desc="Valid URL to the position's official job posting (missing or invalid)",
            parent=basic_info_node,
            critical=True
        )

    # Position Title (must be Dean, not Associate/Assistant/Acting)
    title_leaf = evaluator.add_leaf(
        id=f"P{position_index+1}_Position_Title",
        desc="The position title explicitly includes 'Dean' as the primary role (not Associate/Assistant/Acting)",
        parent=basic_info_node,
        critical=True
    )
    if is_valid_http_url(pos.job_url):
        title_claim = "The job title on this page is a full 'Dean' role and does NOT include or indicate Associate Dean, Assistant Dean, Acting/Interim Dean, or any subordinate dean title."
        if pos.position_title:
            title_claim = f"The job title on this page is '{pos.position_title}', and it is a full 'Dean' role (not Associate/Assistant/Acting/Interim)."
        await evaluator.verify(
            claim=title_claim,
            node=title_leaf,
            sources=pos.job_url,
            additional_instruction="Confirm the title denotes the top dean (e.g., 'Dean, College of X'). Titles containing qualifiers like 'Associate', 'Assistant', 'Interim', or 'Acting' should fail."
        )
    else:
        # Auto-skip handled by prerequisites if URL failed; but keep a verification attempt consistent
        await evaluator.verify(
            claim="The job title is a full 'Dean' role (not Associate/Assistant/Acting).",
            node=title_leaf,
            sources=None,
            additional_instruction="Without a valid posting URL, treat this as unsupported."
        )

    # College Type (Business/Education/Arts & Sciences or equivalent)
    college_leaf = evaluator.add_leaf(
        id=f"P{position_index+1}_College_Type",
        desc="The dean position is specifically for College of Business, College of Education, or College of Arts & Sciences (or equivalent naming)",
        parent=basic_info_node,
        critical=True
    )
    if is_valid_http_url(pos.job_url):
        college_claim = "This posting is for a Dean of a Business, Education, or Arts & Sciences unit (College/School or equivalent, e.g., 'School of Business', 'College of Liberal Arts and Sciences')."
        if pos.college_school:
            college_claim = f"This posting is for Dean of '{pos.college_school}', which is an equivalent to Business, Education, or Arts & Sciences."
        await evaluator.verify(
            claim=college_claim,
            node=college_leaf,
            sources=pos.job_url,
            additional_instruction="Accept naming variations and equivalents: Business (e.g., School of Business, Commerce, Management), Education (e.g., College of Education and Human Development), Arts & Sciences (e.g., Liberal Arts and Sciences, Arts and Sciences). Reject unrelated units like Engineering, Medicine, Law, Health Professions unless explicitly within one of the allowed categories."
        )
    else:
        await evaluator.verify(
            claim="The posting is for a Dean of Business, Education, or Arts & Sciences (or equivalent).",
            node=college_leaf,
            sources=None,
            additional_instruction="Without a valid posting URL, treat this as unsupported."
        )

    # Institution Details
    inst_node = evaluator.add_parallel(
        id=f"P{position_index+1}_Institution_Details",
        desc="University identification and location verification",
        parent=pv_node,
        critical=True
    )

    # University Name
    uni_leaf = evaluator.add_leaf(
        id=f"P{position_index+1}_University_Name",
        desc="The name of the university is clearly identified in the job posting",
        parent=inst_node,
        critical=True
    )
    if is_valid_http_url(pos.job_url):
        uni_claim = "The job posting clearly identifies the hiring university by name."
        if pos.university_name:
            uni_claim = f"The job posting clearly identifies the hiring university as '{pos.university_name}'."
        await evaluator.verify(
            claim=uni_claim,
            node=uni_leaf,
            sources=pos.job_url,
            additional_instruction="Pass if the page names the university/institution prominently (allow reasonable variants like 'The University of X' vs 'X University')."
        )
    else:
        await evaluator.verify(
            claim="The job posting clearly identifies the hiring university by name.",
            node=uni_leaf,
            sources=None,
            additional_instruction="Without a valid posting URL, treat this as unsupported."
        )

    # State Location (must be OH/PA/TN)
    state_leaf = evaluator.add_leaf(
        id=f"P{position_index+1}_State_Location",
        desc="The university is physically located in Ohio, Pennsylvania, or Tennessee",
        parent=inst_node,
        critical=True
    )
    if is_valid_http_url(pos.job_url):
        nstate = normalize_state_name(pos.state) if pos.state else None
        if nstate in ALLOWED_STATES_FULL:
            state_claim = f"The university for this posting is located in {nstate}."
        else:
            state_claim = "The university for this posting is located in one of the following states: Ohio, Pennsylvania, or Tennessee."
        await evaluator.verify(
            claim=state_claim,
            node=state_leaf,
            sources=pos.job_url,
            additional_instruction="Look for campus address or location cues on the page. Pass only if Ohio (OH), Pennsylvania (PA), or Tennessee (TN) is clearly indicated. If location is absent/ambiguous, fail."
        )
    else:
        await evaluator.verify(
            claim="The university is in Ohio, Pennsylvania, or Tennessee.",
            node=state_leaf,
            sources=None,
            additional_instruction="Without a valid posting URL, treat this as unsupported."
        )

    # ----------------- Requirements Details (Qualifications + Search) -------------
    req_node = evaluator.add_parallel(
        id=f"P{position_index+1}_Requirements_Details",
        desc="Verify position requirements and search structure",
        parent=pos_node,
        critical=True
    )

    # Qualifications
    qual_node = evaluator.add_parallel(
        id=f"P{position_index+1}_Qualifications",
        desc="Required qualifications verification",
        parent=req_node,
        critical=True
    )

    # Terminal/Doctoral Degree Required
    degree_leaf = evaluator.add_leaf(
        id=f"P{position_index+1}_Terminal_Degree_Required",
        desc="The job posting explicitly states that a terminal or doctoral degree (Ph.D., Ed.D., or equivalent) is required",
        parent=qual_node,
        critical=True
    )
    if is_valid_http_url(pos.job_url):
        await evaluator.verify(
            claim="The posting explicitly requires a terminal or doctoral degree (e.g., Ph.D., Ed.D., or equivalent) as a qualification (not merely preferred).",
            node=degree_leaf,
            sources=pos.job_url,
            additional_instruction="Look for phrases like 'terminal degree required', 'earned doctorate required', 'Ph.D./Ed.D. required'. If only 'preferred' or ambiguous, fail."
        )
    else:
        await evaluator.verify(
            claim="The posting explicitly requires a terminal or doctoral degree.",
            node=degree_leaf,
            sources=None,
            additional_instruction="Without a valid posting URL, treat this as unsupported."
        )

    # Reporting Structure
    report_leaf = evaluator.add_leaf(
        id=f"P{position_index+1}_Reporting_Structure",
        desc="The dean reports directly to the Provost, Senior Vice President for Academic Affairs, or equivalent chief academic officer",
        parent=qual_node,
        critical=True
    )
    if is_valid_http_url(pos.job_url):
        await evaluator.verify(
            claim="The posting states that the dean reports directly to the Provost, Senior Vice President for Academic Affairs, Chief Academic Officer, or an equivalent CAO title.",
            node=report_leaf,
            sources=pos.job_url,
            additional_instruction="Accept synonyms like 'Executive Vice President and Provost', 'SVP for Academic Affairs', 'Chief Academic Officer'. If reporting line is to President/Chancellor without indicating a CAO, fail."
        )
    else:
        await evaluator.verify(
            claim="The dean reports directly to the Provost or equivalent CAO.",
            node=report_leaf,
            sources=None,
            additional_instruction="Without a valid posting URL, treat this as unsupported."
        )

    # Search Process
    search_node = evaluator.add_parallel(
        id=f"P{position_index+1}_Search_Process",
        desc="Search management and timeline verification",
        parent=req_node,
        critical=True
    )

    # Professional Search Firm
    firm_leaf = evaluator.add_leaf(
        id=f"P{position_index+1}_Professional_Search_Firm",
        desc="The position search is managed by or in partnership with a professional academic search firm or search consultant",
        parent=search_node,
        critical=True
    )
    if is_valid_http_url(pos.job_url):
        firm_claim = "This search is managed by or in partnership with a professional academic search firm or named search consultant."
        if pos.search_firm:
            firm_claim = f"This search is managed by or in partnership with '{pos.search_firm}', a professional academic search firm/consultant."
        await evaluator.verify(
            claim=firm_claim,
            node=firm_leaf,
            sources=pos.job_url,
            additional_instruction="Pass if the page is hosted by a recognized search firm (e.g., WittKieffer, Storbeck, Academic Search, Isaacson Miller, RPA Inc., Greenwood/Asher) or explicitly names such a firm/consultant as managing/assisting the search. Pure university HR postings without a firm should fail."
        )
    else:
        await evaluator.verify(
            claim="The search is managed by a professional academic search firm.",
            node=firm_leaf,
            sources=None,
            additional_instruction="Without a valid posting URL, treat this as unsupported."
        )

    # Current Opening (Posting Date within window)
    date_leaf = evaluator.add_leaf(
        id=f"P{position_index+1}_Current_Opening",
        desc="The position was posted or announced between November 2025 and February 2026, indicating it is a current active search",
        parent=search_node,
        critical=True
    )
    if is_valid_http_url(pos.job_url):
        date_claim = (
            "The posting date shown on this page falls between Nov 1, 2025 and Feb 28, 2026 (inclusive)."
        )
        if pos.posted_date:
            date_claim = (
                f"The posting date on this page is '{pos.posted_date}', and it falls between Nov 1, 2025 and Feb 28, 2026 (inclusive)."
            )
        await evaluator.verify(
            claim=date_claim,
            node=date_leaf,
            sources=pos.job_url,
            additional_instruction="Check for 'Posted', 'Publication', 'Announcement', or similar date. If the date is within 2025-11-01 to 2026-02-28 inclusive, pass. If no date is shown or outside range, fail. 'Updated' dates can count if clearly indicating posting/announcement recency."
        )
    else:
        await evaluator.verify(
            claim="The posting date falls between Nov 1, 2025 and Feb 28, 2026.",
            node=date_leaf,
            sources=None,
            additional_instruction="Without a valid posting URL, treat this as unsupported."
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
    Evaluate an answer for the dean positions task and return a structured result dictionary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Positions are independent
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

    # Extract structured positions
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="extracted_positions"
    )

    # Normalize states in extracted data
    normalized_positions: List[PositionItem] = []
    for p in extracted.positions:
        np = PositionItem(**p.dict())
        np.state = normalize_state_name(np.state)
        normalized_positions.append(np)

    # Keep first 3; pad if fewer
    positions = normalized_positions[:3]
    while len(positions) < 3:
        positions.append(PositionItem())

    # Add custom info for allowed states and time window
    evaluator.add_custom_info(
        {
            "allowed_states_full": list(ALLOWED_STATES_FULL.keys()),
            "allowed_states_abbrev": list(ALLOWED_STATE_ABBR.keys()),
            "time_window_start": TIME_WINDOW_START,
            "time_window_end": TIME_WINDOW_END
        },
        info_type="constraints",
        info_name="allowed_constraints"
    )

    # Build and verify each position subtree
    for idx in range(3):
        await verify_single_position(evaluator, root, positions[idx], idx)

    # Return evaluation summary
    return evaluator.get_summary()