import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mn_superintendent_20260310"
TASK_DESCRIPTION = (
    "As of March 10, 2026, identify a school superintendent position in Minnesota that meets ALL of the following "
    "criteria: (1) The application deadline is on or after March 10, 2026; (2) The school district has an enrollment of "
    "at least 5,000 students; (3) The minimum starting salary offered is at least $240,000; (4) The position requires "
    "candidates to hold a master's degree; (5) The contract length is negotiable in accordance with Minnesota state "
    "statute; (6) The position description emphasizes equity-focused leadership as a key qualification for candidates; "
    "(7) The position is currently accepting applications as of March 10, 2026. Provide the name of the school district, "
    "the application deadline, the district's student enrollment, and the salary range for the position."
)

AS_OF_DATE_STR = "March 10, 2026"
AS_OF_DATE_ISO = "2026-03-10"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionExtraction(BaseModel):
    district_name: Optional[str] = None
    job_title: Optional[str] = None
    application_deadline: Optional[str] = None  # keep as free text (e.g., 'Open until filled' or a date string)
    enrollment: Optional[str] = None            # free text as presented in the answer
    salary_min: Optional[str] = None            # free text or numeric-like string (e.g., '240000', '$240,000')
    salary_max: Optional[str] = None            # free text or numeric-like string
    salary_range_text: Optional[str] = None     # original text snippet for salary range
    job_posting_urls: List[str] = Field(default_factory=list)  # primary posting/application URLs cited
    sources: List[str] = Field(default_factory=list)           # other URLs cited for this position


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_position() -> str:
    return f"""
Extract exactly one superintendent position that the answer proposes for Minnesota. If the answer lists multiple positions,
choose the first one that is explicitly located in Minnesota (MN); if none explicitly say Minnesota, choose the first one.

Return a JSON object with these fields (use null for any missing value):
- district_name: The school district's official name for the selected position.
- job_title: The job title as written in the answer (e.g., "Superintendent" or "Superintendent of Schools").
- application_deadline: The application deadline text as shown (e.g., "March 25, 2026" or "Open until filled").
- enrollment: The district's total student enrollment as stated in the answer (keep exact wording).
- salary_min: The minimum value of the salary if a numeric lower bound is given; otherwise null.
- salary_max: The maximum value of the salary if a numeric upper bound is given; otherwise null.
- salary_range_text: The original salary description text exactly as written in the answer (e.g., "$240,000–$275,000", "not less than $240,000", "commensurate with experience").
- job_posting_urls: An array of the job posting / application URLs cited in the answer for this position.
- sources: An array of any other URLs cited in the answer that substantively support this position (e.g., district profile, news release).
Do not invent any URLs. Extract only URLs explicitly present in the answer; include full protocol (http/https).

Special rules:
- Do NOT parse external webpages here; only extract from the provided answer text.
- For URLs, if duplicates appear across job_posting_urls and sources, still include them in their respective arrays.
- Keep all fields as strings (do not coerce to numbers).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_sanitized_urls(urls: List[str]) -> List[str]:
    """Return unique, sanitized URLs (basic)."""
    seen = set()
    out: List[str] = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Basic normalization: ensure protocol present is already requested in extractor prompt.
        # Accept http(s) only.
        if not (u.startswith("http://") or u.startswith("https://")):
            continue
        # remove trivial trailing slash duplication sensitivity
        key = u[:-1] if u.endswith("/") else u
        if key not in seen:
            seen.add(key)
            out.append(u)
    return out


def _gather_all_sources(extracted: PositionExtraction) -> List[str]:
    urls = (extracted.job_posting_urls or []) + (extracted.sources or [])
    return _unique_sanitized_urls(urls)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_superintendent_position(
    evaluator: Evaluator,
    parent_node,
    extracted: PositionExtraction
) -> None:
    """
    Build the verification subtree for the superintendent position criteria (all critical).
    """
    # Parent node (critical, parallel aggregation)
    position_node = evaluator.add_parallel(
        id="Superintendent_Position_Identification",
        desc="Identify a superintendent position that meets all specified criteria",
        parent=parent_node,
        critical=True
    )

    # Prepare sources
    combined_sources = _gather_all_sources(extracted)
    has_sources = len(combined_sources) > 0
    district_label = extracted.district_name or "the specified district"

    # Create all leaf nodes under the critical parent (all leaves are critical)
    geo_node = evaluator.add_leaf(
        id="Geographic_Location",
        desc="The position is located in Minnesota",
        parent=position_node,
        critical=True
    )
    deadline_node = evaluator.add_leaf(
        id="Application_Deadline",
        desc=f"The application deadline is on or after {AS_OF_DATE_STR}",
        parent=position_node,
        critical=True
    )
    enrollment_node = evaluator.add_leaf(
        id="District_Enrollment",
        desc="The district has an enrollment of at least 5,000 students",
        parent=position_node,
        critical=True
    )
    salary_node = evaluator.add_leaf(
        id="Minimum_Salary",
        desc="The minimum starting salary is at least $240,000",
        parent=position_node,
        critical=True
    )
    masters_node = evaluator.add_leaf(
        id="Master_Degree_Requirement",
        desc="The position requires a master's degree",
        parent=position_node,
        critical=True
    )
    contract_node = evaluator.add_leaf(
        id="Contract_Negotiability",
        desc="The contract length is negotiable according to state statute",
        parent=position_node,
        critical=True
    )
    equity_node = evaluator.add_leaf(
        id="Equity_Leadership_Focus",
        desc="The position emphasizes equity-focused leadership as a key qualification",
        parent=position_node,
        critical=True
    )
    status_node = evaluator.add_leaf(
        id="Application_Status",
        desc=f"The position is accepting applications as of {AS_OF_DATE_STR}",
        parent=position_node,
        critical=True
    )

    # If no sources are available, fail all factual checks immediately (source-grounding policy)
    if not has_sources:
        for n in [
            geo_node, deadline_node, enrollment_node, salary_node,
            masters_node, contract_node, equity_node, status_node
        ]:
            n.score = 0.0
            n.status = "failed"
        evaluator.add_custom_info(
            info={
                "reason": "No source URLs provided in the answer to support the factual claims.",
                "affected_nodes": [
                    "Geographic_Location", "Application_Deadline", "District_Enrollment",
                    "Minimum_Salary", "Master_Degree_Requirement", "Contract_Negotiability",
                    "Equity_Leadership_Focus", "Application_Status"
                ]
            },
            info_type="missing_sources",
            info_name="source_grounding_issue"
        )
        return

    # Build claims and additional instructions
    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    claims_and_sources.append((
        f"The superintendent position for {district_label} is located in Minnesota (MN), United States.",
        combined_sources,
        geo_node,
        "Use the provided URL(s) to confirm the position/district is in Minnesota. Accept 'Minnesota' or 'MN' in the "
        "address, header/footer, or page content. If pages show another state or no evidence of Minnesota, mark as not supported."
    ))

    claims_and_sources.append((
        f"The application deadline for the superintendent position is on or after {AS_OF_DATE_STR}.",
        combined_sources,
        deadline_node,
        f"From the posting/application page, identify the deadline date. Mark CORRECT only if the deadline is {AS_OF_DATE_STR} "
        f"({AS_OF_DATE_ISO}) or later, or if the posting explicitly states 'Open until filled' (or equivalent). If the deadline is "
        f"earlier than {AS_OF_DATE_STR} or the posting indicates the search is closed, mark INCORRECT."
    ))

    claims_and_sources.append((
        "The school district's total student enrollment is at least 5,000 students.",
        combined_sources,
        enrollment_node,
        "Locate the most relevant district-wide enrollment figure on the provided pages (posting, district profile/about). "
        "Treat 'about', 'over', 'approximately' 5,000+ as >= 5,000. If multiple numbers exist, prefer the most official/overall total. "
        "If evidence shows fewer than 5,000 OR enrollment info is not present, mark INCORRECT."
    ))

    claims_and_sources.append((
        "The minimum starting salary for the superintendent is at least $240,000 per year.",
        combined_sources,
        salary_node,
        "Verify salary language on the posting. If a range 'X–Y' is given, evaluate the lower bound X. Accept 'not less than $240,000', "
        "'starting at $240,000', or '$240,000 minimum'. If salary is stated only as 'commensurate' or 'competitive' with no numbers, mark INCORRECT."
    ))

    claims_and_sources.append((
        "The job posting requires that candidates hold at least a master's degree.",
        combined_sources,
        masters_node,
        "Look for 'Master’s degree required' or equivalent wording. Accept synonyms (Master's, MA, MS, M.Ed., EdM). "
        "If it only states 'preferred' (without 'required'), mark INCORRECT."
    ))

    claims_and_sources.append((
        "The superintendent contract term/length is negotiable and the posting references compliance with Minnesota state statute.",
        combined_sources,
        contract_node,
        "PASS only if BOTH are present in the evidence: (1) the contract term/length is stated as 'negotiable' (or equivalent), and "
        "(2) there is an explicit reference to Minnesota statute(s) (e.g., 'Minnesota Statute 123B.143' or 'per MN statutes'). "
        "If either is missing, mark INCORRECT."
    ))

    claims_and_sources.append((
        "The position description emphasizes equity-focused leadership as a key qualification for candidates.",
        combined_sources,
        equity_node,
        "Look for explicit emphasis on equity-focused leadership (e.g., 'equity', 'educational equity', 'equitable outcomes', "
        "'DEI', 'culturally responsive leadership', 'eliminating disparities') in the qualifications/leadership profile. "
        "Generic EEO boilerplate alone does NOT satisfy this."
    ))

    claims_and_sources.append((
        f"As of {AS_OF_DATE_STR}, the position is currently accepting applications.",
        combined_sources,
        status_node,
        f"PASS if the posting indicates 'Open until filled' OR shows a deadline on/after {AS_OF_DATE_STR} OR an active application link. "
        f"If the page indicates 'position filled'/'closed' or the deadline is before {AS_OF_DATE_STR}, mark INCORRECT."
    ))

    # Run all verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
) -> Dict:
    """
    Evaluate an answer for the Minnesota superintendent position task.
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
        default_model=model
    )

    # Record as-of date for transparency
    evaluator.add_custom_info(
        info={"as_of_date_str": AS_OF_DATE_STR, "as_of_date_iso": AS_OF_DATE_ISO},
        info_type="as_of_date"
    )

    # Extract position info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_position(),
        template_class=PositionExtraction,
        extraction_name="position_extraction"
    )

    # Build and run verification subtree
    await verify_superintendent_position(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()