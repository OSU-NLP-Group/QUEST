import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pasa_superintendent_montgomery_feb2026"
TASK_DESCRIPTION = (
    "Identify a superintendent position currently posted on the Pennsylvania Association of School Administrators "
    "(PASA) official website (pasa-net.org) that meets the following criteria: the position was posted in February 2026, "
    "has an application deadline in March 2026, and is for a school district located in Montgomery County, Pennsylvania. "
    "For the identified position, provide the following information: (1) The name of the school district seeking a superintendent, "
    "(2) A direct URL to the PASA job posting page where this position is listed, (3) The county where the school district is located, "
    "(4) The exact application deadline date, (5) The date when the position was posted, (6) Two specific qualification requirements "
    "stated in the job posting: (a) the required educational credential or certification, and (b) any preferred educational degree mentioned. "
    "All information must be verified from official sources, and you must provide reference URLs for the PASA posting page and any district-related information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PASAJobPostingExtraction(BaseModel):
    district_name: Optional[str] = None
    pasa_posting_url: Optional[str] = None
    county: Optional[str] = None
    posting_date: Optional[str] = None
    application_deadline: Optional[str] = None
    required_credential_or_certification: Optional[str] = None
    preferred_degree: Optional[str] = None
    district_official_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pasa_posting() -> str:
    return """
    From the provided answer, extract details for exactly one superintendent position that the answer claims meets
    the following constraints: (a) posted in February 2026, (b) application deadline in March 2026, and
    (c) the district is located in Montgomery County, Pennsylvania.

    Extract the following fields:
    1. district_name: The exact name of the school district as stated in the PASA posting.
    2. pasa_posting_url: A direct URL on pasa-net.org to the specific PASA job posting page where this position is listed.
    3. county: The county where the district is located, as stated in the answer (e.g., "Montgomery County").
    4. posting_date: The exact posting date string for the position (e.g., "February 12, 2026").
    5. application_deadline: The exact application deadline date string (e.g., "March 7, 2026").
    6. required_credential_or_certification: Quote or summarize the required educational credential(s) or certification(s) stated in the posting (e.g., "Pennsylvania Superintendent Letter of Eligibility").
    7. preferred_degree: Quote or summarize any preferred educational degree mentioned (e.g., "Doctorate preferred").
    8. district_official_url: A direct URL to the district's official website page cited in the answer (if present). If none is provided, return null.
    9. additional_urls: Any other official source URLs cited in the answer for this position or district (excluding the PASA posting URL). Return an array; if none, return an empty array.

    IMPORTANT:
    - Extract ONLY what is explicitly present in the answer. Do not invent or infer any values.
    - For URL fields, extract the actual URLs provided in the answer (plain or markdown).
    - If multiple positions are mentioned, select the first one that appears to meet the constraints.
    - If any required field is missing in the answer, return null for that field (or empty array for additional_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _collect_county_sources(extraction: PASAJobPostingExtraction) -> List[str]:
    urls: List[Optional[str]] = [
        extraction.pasa_posting_url,
        extraction.district_official_url,
    ]
    urls.extend(extraction.additional_urls or [])
    return _dedup_urls(urls)


def _safe(value: Optional[str]) -> str:
    return (value or "").strip()


# --------------------------------------------------------------------------- #
# Verification building                                                       #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, info: PASAJobPostingExtraction) -> None:
    # Create the main critical node as per rubric
    task_main = evaluator.add_parallel(
        id="identify_and_report_pasa_superintendent_posting",
        desc="Identify one superintendent position currently posted on PASA (pasa-net.org) that matches the specified timing and location constraints, and report all required fields with official-source URL references.",
        parent=root_node,
        critical=True,
    )

    district_name = _safe(info.district_name)
    pasa_url = _safe(info.pasa_posting_url)
    county_val = _safe(info.county)
    posting_date = _safe(info.posting_date)
    deadline_date = _safe(info.application_deadline)
    required_text = _safe(info.required_credential_or_certification)
    preferred_text = _safe(info.preferred_degree)

    # 1) currently_posted_on_pasa
    node_currently = evaluator.add_leaf(
        id="currently_posted_on_pasa",
        desc="The identified position is a superintendent position currently listed on the PASA official website (pasa-net.org).",
        parent=task_main,
        critical=True,
    )
    claim_currently = "This page on PASA (pasa-net.org) lists a superintendent position vacancy."
    await evaluator.verify(
        claim=claim_currently,
        node=node_currently,
        sources=pasa_url,
        additional_instruction="Verify that the PASA page is a job posting and that it clearly indicates the role is 'Superintendent' (or equivalent variations like 'School District Superintendent')."
    )

    # 2) pasa_posting_url
    node_pasa_url = evaluator.add_leaf(
        id="pasa_posting_url",
        desc="Provides a direct URL on pasa-net.org to the PASA page where the superintendent position is listed.",
        parent=task_main,
        critical=True,
    )
    claim_pasa_url = "The provided URL is hosted on pasa-net.org and directly loads the PASA job posting page for this superintendent position."
    await evaluator.verify(
        claim=claim_pasa_url,
        node=node_pasa_url,
        sources=pasa_url,
        additional_instruction="Check domain is 'pasa-net.org' and the page content is a job posting that includes the superintendent position details."
    )

    # 3) school_district_name_with_citation
    node_district = evaluator.add_leaf(
        id="school_district_name_with_citation",
        desc="States the name of the school district seeking a superintendent as shown on the PASA posting and provides an official-source URL citation.",
        parent=task_main,
        critical=True,
    )
    claim_district = f"The PASA page identifies the school district name as '{district_name}'."
    await evaluator.verify(
        claim=claim_district,
        node=node_district,
        sources=pasa_url,
        additional_instruction="Locate the district name on the PASA posting page; allow minor name variants (e.g., 'SD' vs 'School District'), but the core district name must match."
    )

    # 4) county_is_montgomery_with_verification
    node_county = evaluator.add_leaf(
        id="county_is_montgomery_with_verification",
        desc="States the county for the district and verifies it is Montgomery County, Pennsylvania, using an official source URL citation.",
        parent=task_main,
        critical=True,
    )
    claim_county = "The school district is located in Montgomery County, Pennsylvania."
    county_sources = _collect_county_sources(info)
    await evaluator.verify(
        claim=claim_county,
        node=node_county,
        sources=county_sources,
        additional_instruction="Use official sources (e.g., PASA posting, district website, Pennsylvania government/education pages) to confirm the district is in Montgomery County, PA."
    )

    # 5) posting_date_feb_2026_exact
    node_posting_date = evaluator.add_leaf(
        id="posting_date_feb_2026_exact",
        desc="Provides the exact posting date and verifies it falls in February 2026, citing an official posting/source URL.",
        parent=task_main,
        critical=True,
    )
    claim_posting_date = f"The PASA page shows the position posting date is '{posting_date}' and it falls in February 2026."
    await evaluator.verify(
        claim=claim_posting_date,
        node=node_posting_date,
        sources=pasa_url,
        additional_instruction="Find the 'Posting Date' (or equivalent field) on the PASA page and verify the month is February 2026. Accept common date formats (e.g., 'Feb 3, 2026', 'February 3, 2026')."
    )

    # 6) application_deadline_mar_2026_exact
    node_deadline = evaluator.add_leaf(
        id="application_deadline_mar_2026_exact",
        desc="Provides the exact application deadline date and verifies it falls in March 2026, citing an official posting/source URL.",
        parent=task_main,
        critical=True,
    )
    claim_deadline = f"The PASA page shows the application deadline is '{deadline_date}' and it falls in March 2026."
    await evaluator.verify(
        claim=claim_deadline,
        node=node_deadline,
        sources=pasa_url,
        additional_instruction="Find 'Application Deadline' (or equivalent field) on the PASA page and verify the month is March 2026. Accept common date formats (e.g., 'Mar 7, 2026', 'March 7, 2026')."
    )

    # 7) Required credentials/certification must include PA Superintendent Letter of Eligibility (or eligibility),
    #    AND indicate that a Master's degree is required. Split into two binary leaves for clarity.
    req_group = evaluator.add_parallel(
        id="required_credential_checks",
        desc="Required credentials/certifications from the posting satisfy the constraints.",
        parent=task_main,
        critical=True,
    )

    node_req_letter = evaluator.add_leaf(
        id="required_letter_of_eligibility_present",
        desc="PASA posting includes requirement of Pennsylvania Superintendent Letter of Eligibility (or eligibility to obtain it).",
        parent=req_group,
        critical=True,
    )
    claim_req_letter = "The PASA posting states a requirement for a Pennsylvania Superintendent Letter of Eligibility or eligibility to obtain it."
    await evaluator.verify(
        claim=claim_req_letter,
        node=node_req_letter,
        sources=pasa_url,
        additional_instruction="Check the Qualifications/Requirements section for phrases like 'Pennsylvania Superintendent Letter of Eligibility' or 'eligible to obtain PA Superintendent Letter of Eligibility'. Minor wording variations are acceptable if the meaning is explicit."
    )

    node_req_masters = evaluator.add_leaf(
        id="masters_degree_required_present",
        desc="PASA posting indicates a Master's degree is required.",
        parent=req_group,
        critical=True,
    )
    claim_req_masters = "The PASA posting indicates that a Master's degree is required."
    await evaluator.verify(
        claim=claim_req_masters,
        node=node_req_masters,
        sources=pasa_url,
        additional_instruction="Check the Qualifications/Requirements section for statements like 'Master's degree required' or 'minimum Master's degree'. Allow reasonable wording variants conveying the same requirement."
    )

    # 8) preferred_degree_from_posting_matches_constraint (Doctorate preferred)
    node_preferred = evaluator.add_leaf(
        id="preferred_degree_from_posting_matches_constraint",
        desc="Reports the preferred educational degree from the posting and it must identify a doctorate, with an official-source URL citation.",
        parent=task_main,
        critical=True,
    )
    claim_preferred = "The PASA posting indicates that a doctorate (doctoral degree) is preferred."
    await evaluator.verify(
        claim=claim_preferred,
        node=node_preferred,
        sources=pasa_url,
        additional_instruction="Look for phrases such as 'Doctorate preferred', 'doctoral degree preferred', 'Ph.D. or Ed.D. preferred', or equivalent wording in the posting."
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
    # Initialize evaluator and root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is a container; main task node below will be critical
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_pasa_posting(),
        template_class=PASAJobPostingExtraction,
        extraction_name="pasa_job_posting_extraction",
    )

    # Optionally record custom info (e.g., constraint summary)
    evaluator.add_custom_info(
        info={
            "constraints": {
                "posting_month_year": "February 2026",
                "deadline_month_year": "March 2026",
                "county_required": "Montgomery County, PA",
            }
        },
        info_type="constraints_summary",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extraction)

    # Return structured result summary
    return evaluator.get_summary()