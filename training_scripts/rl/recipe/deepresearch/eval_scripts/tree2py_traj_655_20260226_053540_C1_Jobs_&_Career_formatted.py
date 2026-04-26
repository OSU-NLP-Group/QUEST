import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ferris_asst_prof_tenure_track_mi_residency"
TASK_DESCRIPTION = (
    "Find a tenure-track Assistant Professor position at Ferris State University that requires a PhD degree and "
    "specifies that the candidate must reside in Michigan after acceptance of employment. Provide the following "
    "information about this position: (1) The position title and department, (2) The anticipated start date as "
    "stated in the job posting, and (3) The direct application link on jobs.ferris.edu."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionInfo(BaseModel):
    position_title: Optional[str] = None
    department: Optional[str] = None
    institution: Optional[str] = None
    anticipated_start_date: Optional[str] = None
    # All URLs mentioned that directly relate to this job posting (including the application page if present)
    job_posting_urls: List[str] = Field(default_factory=list)
    # The direct application link that the answer claims is on jobs.ferris.edu
    application_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_position() -> str:
    return """
Extract the following structured information for the single Ferris State University position described in the answer:

Required fields:
- position_title: The full position title as presented in the answer (e.g., "Assistant Professor of X").
- department: The department/program/unit name as presented (e.g., "Department of Y", "School of Z", or "College of ...").
- institution: The institution name explicitly mentioned (should be "Ferris State University" if present).
- anticipated_start_date: The anticipated/expected start date string exactly as stated in the answer.
- job_posting_urls: An array of all URLs in the answer that directly point to the job posting or related official job details pages (include jobs.ferris.edu links and any other official posting links mentioned).
- application_url: The direct application link on jobs.ferris.edu if provided in the answer. If multiple jobs.ferris.edu URLs are present, choose the one that appears to be the apply/postings page for this specific job. If not provided, set to null.

Important:
- Extract only what is explicitly stated in the answer. Do not invent missing values.
- For URLs, extract the actual URL strings (from plain text or markdown links).
- If any field is missing in the answer, set it to null (for strings) or [] (for arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _combine_sources(info: PositionInfo) -> List[str]:
    urls: List[str] = []
    if info.application_url:
        urls.append(info.application_url)
    if info.job_posting_urls:
        urls.extend(info.job_posting_urls)
    return _dedup_urls(urls)


def _is_jobs_ferris_domain(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        return "jobs.ferris.edu" in host
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, info: PositionInfo) -> None:
    """
    Build the verification tree according to the rubric and run all checks.
    """
    # Root wrapper node (framework's root is always non-critical). Create a critical top-level node per rubric.
    top_node = evaluator.add_parallel(
        id="Ferris_State_Assistant_Professor_Position",
        desc="Evaluate whether the identified position is a valid tenure-track Assistant Professor role at Ferris State University with required qualifications and application information",
        parent=evaluator.root,
        critical=True,
    )

    # Prepare sources once
    all_sources = _combine_sources(info)

    # ------------------------------ Group 1 -------------------------------- #
    # Position Type and Institution (critical)
    pos_node = evaluator.add_parallel(
        id="Position_Type_and_Institution",
        desc="The position must be a tenure-track Assistant Professor position at Ferris State University",
        parent=top_node,
        critical=True,
    )

    # Title sub-sequence: require title provided, then verify on page
    title_seq = evaluator.add_sequential(
        id="Position_Title_Info",
        desc="Position title is provided and supported by sources",
        parent=pos_node,
        critical=True,
    )
    title_provided = evaluator.add_custom_node(
        result=bool(info.position_title and info.position_title.strip()),
        id="title_provided",
        desc="Position title is provided in the answer",
        parent=title_seq,
        critical=True,
    )
    title_supported = evaluator.add_leaf(
        id="title_supported_by_sources",
        desc="Position title matches what appears on the cited job posting page(s)",
        parent=title_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The job posting lists the position title as '{info.position_title or ''}'.",
        node=title_supported,
        sources=all_sources,
        additional_instruction="Allow reasonable punctuation/casing variants. If the posting lists a longer title containing this phrase, consider it a match.",
    )

    # Department sub-sequence: require department provided, then verify on page
    dept_seq = evaluator.add_sequential(
        id="Department_Info",
        desc="Department is provided and supported by sources",
        parent=pos_node,
        critical=True,
    )
    department_provided = evaluator.add_custom_node(
        result=bool(info.department and info.department.strip()),
        id="department_provided",
        desc="Department is provided in the answer",
        parent=dept_seq,
        critical=True,
    )
    department_supported = evaluator.add_leaf(
        id="department_supported_by_sources",
        desc="Department matches what appears on the cited job posting page(s)",
        parent=dept_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The job posting indicates the department/unit/program as '{info.department or ''}'.",
        node=department_supported,
        sources=all_sources,
        additional_instruction="Accept close variants such as 'Department of', 'School of', or 'College of' if they reasonably refer to the same unit.",
    )

    # Assistant Professor check
    asst_prof_leaf = evaluator.add_leaf(
        id="assistant_professor_verified",
        desc="The position is for an Assistant Professor role",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This job posting is for an Assistant Professor position (Assistant Professor may appear as 'Assistant Professor of ...' or similar).",
        node=asst_prof_leaf,
        sources=all_sources,
        additional_instruction="Allow reasonable variants like 'Asst. Professor'. If multiple ranks are listed (e.g., Assistant/Associate), it still satisfies as long as Assistant level is included.",
    )

    # Tenure-track check
    tenure_track_leaf = evaluator.add_leaf(
        id="tenure_track_verified",
        desc="The position is tenure-track",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The position is tenure-track (may be written as 'tenure track', 'tenure-track', 'tenure earning', or similar).",
        node=tenure_track_leaf,
        sources=all_sources,
        additional_instruction="Only pass if tenure-track (or equivalent) is clearly indicated. 'Non-tenure' or 'term/temporary' should not pass.",
    )

    # Institution check
    institution_leaf = evaluator.add_leaf(
        id="institution_verified",
        desc="The job is at Ferris State University",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This job posting is for a position at Ferris State University.",
        node=institution_leaf,
        sources=all_sources,
        additional_instruction="It should explicitly indicate 'Ferris State University' or 'Ferris State'.",
    )

    # ------------------------------ Group 2 -------------------------------- #
    # Required Qualifications (critical)
    req_node = evaluator.add_parallel(
        id="Required_Qualifications",
        desc="The position must require a PhD degree and specify that the candidate must reside in Michigan after acceptance of employment",
        parent=top_node,
        critical=True,
    )

    # PhD required
    phd_leaf = evaluator.add_leaf(
        id="phd_required_verified",
        desc="The posting requires a PhD (or equivalent doctoral degree)",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting states that a PhD (or doctoral degree) is required for this position. 'Preferred' alone is not sufficient; 'required by start date' counts.",
        node=phd_leaf,
        sources=all_sources,
        additional_instruction="Look for phrases like 'PhD required', 'earned doctorate required', or 'PhD by start date required'. If it only says 'preferred', this should fail.",
    )

    # Michigan residency requirement after acceptance
    residency_leaf = evaluator.add_leaf(
        id="mi_residency_requirement_verified",
        desc="The posting specifies that the candidate must reside in Michigan after acceptance of employment",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting specifies that the candidate must reside in Michigan after acceptance of employment (e.g., immediately or within a listed timeframe).",
        node=residency_leaf,
        sources=all_sources,
        additional_instruction="Allow variants such as 'must reside in Michigan within X days/weeks/months after accepting the position' or 'must move to/maintain residence in Michigan after acceptance'.",
    )

    # ------------------------------ Group 3 -------------------------------- #
    # Application Details (critical)
    app_node = evaluator.add_parallel(
        id="Application_Details",
        desc="The position must include an anticipated start date and provide a direct application link on jobs.ferris.edu",
        parent=top_node,
        critical=True,
    )

    # Start date sequence: must be provided then supported
    start_seq = evaluator.add_sequential(
        id="Start_Date_Info",
        desc="Anticipated start date is provided and supported by sources",
        parent=app_node,
        critical=True,
    )
    start_provided = evaluator.add_custom_node(
        result=bool(info.anticipated_start_date and info.anticipated_start_date.strip()),
        id="start_date_provided",
        desc="Anticipated start date is provided in the answer",
        parent=start_seq,
        critical=True,
    )
    start_supported = evaluator.add_leaf(
        id="start_date_supported_by_sources",
        desc="Anticipated start date matches what appears on the cited job posting page(s)",
        parent=start_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The job posting lists the anticipated/expected start date as '{info.anticipated_start_date or ''}'.",
        node=start_supported,
        sources=all_sources,
        additional_instruction="Accept close paraphrases like 'anticipated start date', 'expected start date', 'position start date', or a concrete date (month/year).",
    )

    # Application URL presence
    app_link_present = evaluator.add_custom_node(
        result=bool(info.application_url and info.application_url.strip()),
        id="application_link_provided",
        desc="Direct application link is provided in the answer",
        parent=app_node,
        critical=True,
    )

    # Application URL domain validation
    app_link_domain_valid = evaluator.add_custom_node(
        result=_is_jobs_ferris_domain(info.application_url),
        id="application_link_on_jobs_ferris_edu",
        desc="Direct application link is on jobs.ferris.edu",
        parent=app_node,
        critical=True,
    )

    # Application URL is the direct application page (verified by URL)
    app_link_direct_leaf = evaluator.add_leaf(
        id="application_link_is_direct",
        desc="The provided URL is the direct application link for the job on jobs.ferris.edu",
        parent=app_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is the direct application page for the position on jobs.ferris.edu (i.e., the page where applicants can apply/submit).",
        node=app_link_direct_leaf,
        sources=info.application_url if info.application_url else None,
        additional_instruction="The page should be an official Ferris jobs page (e.g., contains 'postings' or shows an Apply button/section). If it's just the general jobs homepage without the specific job, this should fail.",
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
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root wrapper; actual rubric root is added under it
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
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_position(),
        template_class=PositionInfo,
        extraction_name="position_info",
    )

    # Build verification tree and run verifications
    await build_verification_tree(evaluator, extracted_info)

    # Return the summary
    return evaluator.get_summary()