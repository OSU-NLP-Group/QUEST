import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "edu_positions_eastern_us"
TASK_DESCRIPTION = """
I am exploring career opportunities in higher education and K-12 education in the Eastern United States. Find three full-time professional job openings currently accepting applications at accredited four-year universities or major public school districts located in states east of the Mississippi River. The positions must represent at least two different job categories (such as faculty, administrative staff, student services, facilities, or technology). For each position, provide: (1) The official position title, (2) The name of the hiring institution, (3) The department or unit, (4) The application deadline or close date, (5) A direct URL to the official job posting on the institution's career portal, and (6) A brief summary of the minimum qualification requirements (education and/or experience). Ensure that all three positions are currently open and accepting applications, and are full-time professional roles (not student positions, part-time, or temporary positions).
"""

# States considered east of the Mississippi River (full state names and common abbreviations)
EASTERN_STATES = {
    "Alabama", "AL",
    "Connecticut", "CT",
    "Delaware", "DE",
    "District of Columbia", "DC",
    "Florida", "FL",
    "Georgia", "GA",
    "Illinois", "IL",
    "Indiana", "IN",
    "Kentucky", "KY",
    "Maine", "ME",
    "Maryland", "MD",
    "Massachusetts", "MA",
    "Michigan", "MI",
    "Mississippi", "MS",
    "New Hampshire", "NH",
    "New Jersey", "NJ",
    "New York", "NY",
    "North Carolina", "NC",
    "Ohio", "OH",
    "Pennsylvania", "PA",
    "Rhode Island", "RI",
    "South Carolina", "SC",
    "Tennessee", "TN",
    "Vermont", "VT",
    "Virginia", "VA",
    "West Virginia", "WV",
    "Wisconsin", "WI",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class JobPosition(BaseModel):
    """One extracted job position."""
    title: Optional[str] = None
    institution_name: Optional[str] = None
    department: Optional[str] = None
    application_deadline: Optional[str] = None  # Accepts date or phrases like "Open until filled"
    posting_url: Optional[str] = None
    qualifications_summary: Optional[str] = None  # Brief minimum requirements summary
    job_category: Optional[str] = None  # e.g., "faculty", "administrative", "student services", "technology", "facilities"
    location_city: Optional[str] = None
    location_state: Optional[str] = None  # Prefer full name or abbreviation
    employment_type_text: Optional[str] = None  # free text (e.g., "Full-Time", "FT", etc.)
    current_status_text: Optional[str] = None  # free text indicating open/accepting
    application_process_summary: Optional[str] = None  # brief description of how to apply


class PositionsExtraction(BaseModel):
    """Model for all positions mentioned in the answer."""
    positions: List[JobPosition] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract all job positions mentioned in the answer. For each position, return the following fields exactly as stated in the answer (do not invent or infer):
    - title: The official position title.
    - institution_name: The hiring university or school district name.
    - department: The department or unit.
    - application_deadline: The application deadline or close date. If not explicitly provided, return null. Accept phrases like "Open until filled" or a specific date.
    - posting_url: The direct URL to the official job posting on the institution's careers site or official website. If missing, return null.
    - qualifications_summary: A brief summary of the minimum qualification requirements (education and/or experience) stated in the answer. If the answer does not provide a summary, return null.
    - job_category: A single category label capturing the type of role from these options only: "faculty", "administrative", "student services", "technology", "facilities", "other". Use what is explicitly stated or clearly implied by the answer; otherwise return "other".
    - location_city: City or locality for the position (if mentioned).
    - location_state: State for the position (full name or 2-letter abbreviation), if mentioned in the answer.
    - employment_type_text: Any text indicating employment type (e.g., "Full-Time", "full time", "FT", etc.) from the answer.
    - current_status_text: Any text indicating that the posting is currently open/accepting applications (e.g., "Apply Now", "accepting applications", "open until filled") from the answer. If not present, return null.
    - application_process_summary: Brief description of the application process (e.g., "Apply via portal", "Submit resume + cover letter") if provided; else return null.

    Return a JSON object with key 'positions' as an array of JobPosition entries.
    The URLs must be explicitly present in the answer. If a URL is missing a protocol, prepend 'http://'.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def states_instruction() -> str:
    """Compose human-readable list of states east of the Mississippi River to guide the verifier."""
    ordered_states = [
        "AL", "CT", "DE", "DC", "FL", "GA", "IL", "IN", "KY", "ME", "MD", "MA", "MI", "MS",
        "NH", "NJ", "NY", "NC", "OH", "PA", "RI", "SC", "TN", "VT", "VA", "WV", "WI"
    ]
    return (
        "A position qualifies if its location is in a state east of the Mississippi River. "
        "Accept any of the following state names or abbreviations on the job page: "
        + ", ".join(ordered_states)
        + ". If the page shows a city (e.g., 'Philadelphia, PA' or 'Chapel Hill, NC'), treat it as valid. "
        "Remote roles should be considered valid only if they explicitly indicate a primary location in one of these states."
    )


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification per-position                                                   #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: JobPosition,
    idx: int,
) -> None:
    """
    Build the verification sub-tree for one position and perform checks.
    """

    # Create container for this position (non-critical, parallel)
    position_node = evaluator.add_parallel(
        id=f"Position_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} job position meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # ----------------- Basic Information (Critical group) ----------------- #
    basic_info_node = evaluator.add_parallel(
        id=f"Position_{idx+1}_Basic_Information",
        desc="Complete basic information about the position is provided",
        parent=position_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(pos.title),
        id=f"Position_{idx+1}_Title",
        desc="Official position title is provided",
        parent=basic_info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(pos.institution_name),
        id=f"Position_{idx+1}_Institution_Name",
        desc="Hiring institution name is provided",
        parent=basic_info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(pos.department),
        id=f"Position_{idx+1}_Department",
        desc="Department or unit is provided",
        parent=basic_info_node,
        critical=True
    )

    # ----------------- Application Details (Critical group) ---------------- #
    app_details_node = evaluator.add_parallel(
        id=f"Position_{idx+1}_Application_Details",
        desc="Complete application information is provided and verifiable",
        parent=position_node,
        critical=True
    )

    # Deadline leaf: verify against posting URL if provided; otherwise fail
    deadline_leaf = evaluator.add_leaf(
        id=f"Position_{idx+1}_Deadline",
        desc="Application deadline or close date is provided",
        parent=app_details_node,
        critical=True
    )
    if _non_empty(pos.application_deadline) and _non_empty(pos.posting_url):
        claim = f"The job posting states the application deadline or close date as: {pos.application_deadline}."
        await evaluator.verify(
            claim=claim,
            node=deadline_leaf,
            sources=pos.posting_url,
            additional_instruction="Accept 'Open until filled' or similar phrasing as a valid close date if present."
        )
    else:
        deadline_leaf.score = 0.0
        deadline_leaf.status = "failed"

    # URL leaf: check URL is accessible and represents the official posting
    url_leaf = evaluator.add_leaf(
        id=f"Position_{idx+1}_URL",
        desc="Direct URL to the official job posting is provided and accessible",
        parent=app_details_node,
        critical=True
    )
    if _non_empty(pos.posting_url):
        claim = "This page is an official job posting on the institution's career portal or official website and is accessible."
        await evaluator.verify(
            claim=claim,
            node=url_leaf,
            sources=pos.posting_url,
            additional_instruction="Treat .edu domains and official district/agency portals (e.g., 'Public Schools', 'School District') as official. The page should show a job posting, not a general info page."
        )
    else:
        url_leaf.score = 0.0
        url_leaf.status = "failed"

    # URL reference leaf: details on page match title and institution
    url_ref_leaf = evaluator.add_leaf(
        id=f"Position_{idx+1}_URL_Reference",
        desc="The provided URL leads to the correct job posting with matching details",
        parent=app_details_node,
        critical=True
    )
    if _non_empty(pos.posting_url) and (_non_empty(pos.title) or _non_empty(pos.institution_name)):
        label_title = pos.title or ""
        label_inst = pos.institution_name or ""
        claim = f"The job posting page lists the job title '{label_title}' and indicates the hiring institution '{label_inst}'."
        await evaluator.verify(
            claim=claim,
            node=url_ref_leaf,
            sources=pos.posting_url,
            additional_instruction="Minor variations in title formatting are acceptable (e.g., punctuation, capitalization). Institution branding or logo counts as indication."
        )
    else:
        url_ref_leaf.score = 0.0
        url_ref_leaf.status = "failed"

    # ----------------- Other Critical leaves on the position node ---------- #
    # Institution validity: university (4-year) or major public school district
    inst_valid_leaf = evaluator.add_leaf(
        id=f"Position_{idx+1}_Institution_Validity",
        desc="The position is at an accredited four-year university or major public school district in the United States",
        parent=position_node,
        critical=True
    )
    claim_inst = (
        f"The hiring institution '{pos.institution_name or 'the institution'}' is a four-year university or a public school district in the United States."
    )
    await evaluator.verify(
        claim=claim_inst,
        node=inst_valid_leaf,
        sources=pos.posting_url if _non_empty(pos.posting_url) else None,
        additional_instruction="Consider '.edu' university sites and pages explicitly referencing 'Public Schools' or 'School District' as valid. You do not need explicit 'accredited' wording; rely on institutional type indicated by the page.",
        extra_prerequisites=[url_leaf] if _non_empty(pos.posting_url) else None
    )

    # Geographic location: must be in state east of Mississippi River
    geo_leaf = evaluator.add_leaf(
        id=f"Position_{idx+1}_Geographic_Location",
        desc="The position is located in a state east of the Mississippi River",
        parent=position_node,
        critical=True
    )
    if _non_empty(pos.location_state):
        claim_geo = (
            f"The job posting indicates the position is located in {pos.location_state}, which is east of the Mississippi River."
        )
    else:
        # Allow verifier to infer from page location text
        claim_geo = "The job posting indicates the position is in a state east of the Mississippi River."
    await evaluator.verify(
        claim=claim_geo,
        node=geo_leaf,
        sources=pos.posting_url if _non_empty(pos.posting_url) else None,
        additional_instruction=states_instruction(),
        extra_prerequisites=[url_leaf] if _non_empty(pos.posting_url) else None
    )

    # Employment type: full-time professional, not student/part-time/temporary
    emp_leaf = evaluator.add_leaf(
        id=f"Position_{idx+1}_Employment_Type",
        desc="The position is a full-time professional role (not student, part-time, or temporary)",
        parent=position_node,
        critical=True
    )
    claim_emp = (
        "This job is a full-time professional position (not a student role, not part-time, and not temporary)."
    )
    await evaluator.verify(
        claim=claim_emp,
        node=emp_leaf,
        sources=pos.posting_url if _non_empty(pos.posting_url) else None,
        additional_instruction="Look for indicators like 'Full-Time', 'FTE 1.0', or similar. Exclude student-only roles, part-time listings, or explicitly temporary/seasonal.",
        extra_prerequisites=[url_leaf] if _non_empty(pos.posting_url) else None
    )

    # Current status: currently posted and accepting applications
    status_leaf = evaluator.add_leaf(
        id=f"Position_{idx+1}_Current_Status",
        desc="The position is currently posted and accepting applications",
        parent=position_node,
        critical=True
    )
    claim_status = (
        "The job posting indicates that applications are currently being accepted."
    )
    await evaluator.verify(
        claim=claim_status,
        node=status_leaf,
        sources=pos.posting_url if _non_empty(pos.posting_url) else None,
        additional_instruction="Indicators include an 'Apply' button/link, 'Accepting applications', 'Open until filled', or a present application portal link. If the page clearly states 'closed' or 'no longer accepting', it should fail.",
        extra_prerequisites=[url_leaf] if _non_empty(pos.posting_url) else None
    )

    # Application process accessibility
    app_proc_leaf = evaluator.add_leaf(
        id=f"Position_{idx+1}_Application_Process",
        desc="The position has a clearly stated application process accessible through the institution's official career portal or website",
        parent=position_node,
        critical=True
    )
    claim_proc = "The job posting page clearly states how to apply or provides an 'Apply' button/link to the institution's application portal."
    await evaluator.verify(
        claim=claim_proc,
        node=app_proc_leaf,
        sources=pos.posting_url if _non_empty(pos.posting_url) else None,
        additional_instruction="Look for phrases like 'Apply', 'Submit application', 'How to apply', or direct portal links.",
        extra_prerequisites=[url_leaf] if _non_empty(pos.posting_url) else None
    )

    # Qualifications summary: supported by the posting
    quals_leaf = evaluator.add_leaf(
        id=f"Position_{idx+1}_Qualifications",
        desc="Minimum qualification requirements are provided",
        parent=position_node,
        critical=True
    )
    if _non_empty(pos.qualifications_summary) and _non_empty(pos.posting_url):
        claim_quals = f"The job posting includes minimum qualifications consistent with: {pos.qualifications_summary}."
        await evaluator.verify(
            claim=claim_quals,
            node=quals_leaf,
            sources=pos.posting_url,
            additional_instruction="Check that the page lists minimum education/experience requirements that reasonably match the provided summary. Minor paraphrasing is acceptable.",
            extra_prerequisites=[url_leaf]
        )
    else:
        # If either summary or URL missing, fail this critical leaf
        quals_leaf.score = 0.0
        quals_leaf.status = "failed"


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
    Evaluate an answer for the Eastern US education jobs task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Positions evaluated independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find three full-time professional job openings at educational institutions in the Eastern United States, providing complete application details for each position",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract positions from the answer
    extracted_positions = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Keep only the first 3 positions (pad with empty placeholders if fewer)
    positions = list(extracted_positions.positions[:3])
    while len(positions) < 3:
        positions.append(JobPosition())

    # Add a small custom info block to facilitate debugging
    evaluator.add_custom_info(
        info={
            "positions_count_in_answer": len(extracted_positions.positions),
            "used_positions_count": 3,
            "categories": [p.job_category for p in positions],
        },
        info_type="extraction_stats",
        info_name="extraction_overview"
    )

    # Verify the three positions
    for i in range(3):
        await verify_position(evaluator, root, positions[i], i)

    # Category diversity (Critical leaf under root)
    # We compute distinct categories (non-empty) from the first 3 positions
    distinct_categories = { (positions[i].job_category or "").strip().lower() for i in range(3) if _non_empty(positions[i].job_category) }
    category_diversity_leaf = evaluator.add_leaf(
        id="Category_Diversity",
        desc="The three positions represent at least two different job categories (e.g., faculty, administrative, student services, facilities, technology)",
        parent=root,
        critical=True
    )
    # Use simple logic check via the verifier for transparency, but also compute and pass as instruction
    diversity_claim = (
        f"The three positions include at least two distinct job categories. Extracted categories: {list(distinct_categories)}."
    )
    await evaluator.verify(
        claim=diversity_claim,
        node=category_diversity_leaf,
        additional_instruction="This is a logical check based on the extracted categories from the answer. Consider categories identical despite minor wording variations (e.g., 'admin' vs 'administrative'). Pass if there are 2 or more distinct categories."
    )

    # Return evaluation summary
    return evaluator.get_summary()