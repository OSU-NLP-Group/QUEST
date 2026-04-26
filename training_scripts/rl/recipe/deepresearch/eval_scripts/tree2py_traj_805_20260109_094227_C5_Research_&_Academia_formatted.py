import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "icml2025_submission_requirements"
TASK_DESCRIPTION = """
A PhD student in machine learning has completed their first research paper and is preparing to submit it to ICML 2025. The paper is 8 pages long (excluding references and appendices), uses PyTorch for implementation, includes experiments on publicly available benchmark datasets, and the student has never reviewed for a major conference before. To ensure a successful submission, the student needs to compile a comprehensive submission requirements document. Identify and provide the following information for ICML 2025: (1) Conference location (city and country) and the specific dates for the main conference sessions; (2) The three critical submission deadlines: abstract submission, full paper submission, and supplementary material submission (specify the dates and timezone); (3) Paper format specifications including the maximum page limit for the main paper body at initial submission, the page limit for the camera-ready version, and the maximum file size for initial submission; (4) The name of the required LaTeX style package and whether any mandatory statements (such as impact statements) must be included in the submission; (5) Whether the conference requires authors to serve as reviewers and specifically what obligations apply to first-time submitters versus authors with 4 or more submissions; (6) The submission platform name and the suggested deadline for creating an account on that platform.
"""


# --------------------------------------------------------------------------- #
# Ground truth (expected facts per rubric)                                    #
# --------------------------------------------------------------------------- #
EXPECTED = {
    "conference_logistics": {
        "location": "Vancouver, Canada",
        "main_conference_dates": "July 13–19, 2025"
    },
    "submission_deadlines": {
        "abstract": "January 23, 2025 (AoE)",
        "full_paper": "January 30, 2025 (AoE)",
        "supplementary": "No separate deadline; due with full paper on January 30, 2025 (AoE)",
        "timezone": "AoE (UTC−12)"
    },
    "paper_format_specifications": {
        "initial_page_limit": "8 pages for main paper body; references/appendices/impact statement do not count",
        "camera_ready_page_limit": "9 pages for main paper body",
        "initial_file_size_limit": "50MB",
        "camera_ready_file_size_limit": "20MB",
        "file_and_typesetting_format": "PDF produced using LaTeX; other typesetting not supported"
    },
    "required_components": {
        "latex_style": "icml2025 (ICML 2025 style files)",
        "impact_statement": "Impact Statement is mandatory and does not count toward page limit",
        "double_blind_anonymization": "Double-blind review; submissions must be anonymized"
    },
    "reviewer_obligations": {
        "at_least_one_reviewer_per_submission": "At least one author per submission must agree to serve as a reviewer, with exemptions (e.g., insufficient qualifications, organizing roles)",
        "four_or_more_submissions_rule": "Authors with 4+ submissions must agree to serve as reviewers, with exemptions (e.g., AC/SAC/organizing roles)",
        "first_time_submitter_applicability": "First-time submitters have no special exception; they must agree to review unless exempt or unqualified"
    },
    "platform_requirements": {
        "platform_name": "OpenReview",
        "suggested_account_creation_deadline": "January 9, 2025"
    }
}


# --------------------------------------------------------------------------- #
# Extraction models                                                            #
# --------------------------------------------------------------------------- #
class LogisticsInfo(BaseModel):
    location: Optional[str] = None
    main_conference_dates: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DeadlinesInfo(BaseModel):
    abstract_deadline: Optional[str] = None
    full_paper_deadline: Optional[str] = None
    supplementary_deadline: Optional[str] = None  # e.g., "same as full paper", or a date
    timezone: Optional[str] = None  # e.g., "AoE" or "Anywhere on Earth"
    sources: List[str] = Field(default_factory=list)


class FormatInfo(BaseModel):
    initial_page_limit: Optional[str] = None
    camera_ready_page_limit: Optional[str] = None
    initial_file_size_limit: Optional[str] = None
    camera_ready_file_size_limit: Optional[str] = None
    file_and_typesetting_format: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RequiredComponentsInfo(BaseModel):
    latex_style: Optional[str] = None
    impact_statement: Optional[str] = None
    double_blind_anonymization: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ReviewerObligationsInfo(BaseModel):
    at_least_one_reviewer_per_submission: Optional[str] = None
    four_or_more_submissions_rule: Optional[str] = None
    first_time_submitter_applicability: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PlatformInfo(BaseModel):
    platform_name: Optional[str] = None
    suggested_account_creation_deadline: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ICML2025Extraction(BaseModel):
    logistics: Optional[LogisticsInfo] = None
    submission_deadlines: Optional[DeadlinesInfo] = None
    paper_format_specifications: Optional[FormatInfo] = None
    required_components: Optional[RequiredComponentsInfo] = None
    reviewer_obligations: Optional[ReviewerObligationsInfo] = None
    platform_requirements: Optional[PlatformInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                            #
# --------------------------------------------------------------------------- #
def prompt_extract_icml2025() -> str:
    return """
Extract the ICML 2025 submission requirements as explicitly stated in the answer. For each group below, return the fields as strings exactly as written in the answer (do not infer or normalize), and collect all URLs cited in the answer that support that group into a 'sources' array of URLs (extract only actual URLs present).

Return a single JSON object with these top-level objects:

1) logistics:
   - location: string or null (e.g., "Vancouver, Canada")
   - main_conference_dates: string or null (e.g., "July 13–19, 2025")
   - sources: array of URL strings (may be empty)

2) submission_deadlines:
   - abstract_deadline: string or null (e.g., "January 23, 2025 AoE")
   - full_paper_deadline: string or null (e.g., "January 30, 2025 AoE")
   - supplementary_deadline: string or null (e.g., "same as full paper" or a date)
   - timezone: string or null (e.g., "AoE", "Anywhere on Earth", or "UTC−12")
   - sources: array of URL strings (may be empty)

3) paper_format_specifications:
   - initial_page_limit: string or null (e.g., "8 pages; refs/appendices/impact do not count")
   - camera_ready_page_limit: string or null (e.g., "9 pages")
   - initial_file_size_limit: string or null (e.g., "50MB")
   - camera_ready_file_size_limit: string or null (e.g., "20MB")
   - file_and_typesetting_format: string or null (e.g., "PDF via LaTeX only")
   - sources: array of URL strings (may be empty)

4) required_components:
   - latex_style: string or null (e.g., "icml2025")
   - impact_statement: string or null (e.g., "mandatory; not counted toward page limit")
   - double_blind_anonymization: string or null (e.g., "double-blind; anonymized")
   - sources: array of URL strings (may be empty)

5) reviewer_obligations:
   - at_least_one_reviewer_per_submission: string or null (e.g., "at least one author must review; exemptions exist")
   - four_or_more_submissions_rule: string or null (e.g., "authors with 4+ submissions must review; exemptions")
   - first_time_submitter_applicability: string or null (e.g., "no special exception for first timers")
   - sources: array of URL strings (may be empty)

6) platform_requirements:
   - platform_name: string or null (e.g., "OpenReview")
   - suggested_account_creation_deadline: string or null (e.g., "January 9, 2025")
   - sources: array of URL strings (may be empty)

Rules:
- Extract only information explicitly present in the answer.
- If a field is missing from the answer, set it to null.
- For 'sources', extract all valid URLs mentioned for that group (include links in markdown).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        v = u.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _collect_all_sources(extracted: ICML2025Extraction) -> List[str]:
    urls: List[str] = []
    if extracted.logistics:
        urls.extend(extracted.logistics.sources)
    if extracted.submission_deadlines:
        urls.extend(extracted.submission_deadlines.sources)
    if extracted.paper_format_specifications:
        urls.extend(extracted.paper_format_specifications.sources)
    if extracted.required_components:
        urls.extend(extracted.required_components.sources)
    if extracted.reviewer_obligations:
        urls.extend(extracted.reviewer_obligations.sources)
    if extracted.platform_requirements:
        urls.extend(extracted.platform_requirements.sources)
    return _dedup_urls(urls)


def _prefer_sources(primary: Optional[List[str]], fallback_all: List[str]) -> List[str]:
    primary = primary or []
    if primary:
        return _dedup_urls(primary)
    return _dedup_urls(fallback_all)


async def _add_and_verify(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    add_ins: Optional[str] = None,
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction=add_ins or (
            "Verify strictly against the provided URLs (official ICML 2025 pages such as Call for Papers, "
            "Author Instructions, Important Dates, OpenReview CFP). Any single supporting URL suffices. "
            "Allow minor wording or formatting variations (e.g., date range punctuation). Treat 'AoE' as 'Anywhere on Earth (UTC−12)'."
        ),
    )


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root,
    extracted: ICML2025Extraction,
):
    # Make root critical to follow rubric (all descendants must be critical)
    root.critical = True

    all_sources = _collect_all_sources(extracted)

    # 1) Conference logistics
    conf_node = evaluator.add_parallel(
        id="conference_logistics",
        desc="Conference location and main conference dates are correctly provided",
        parent=root,
        critical=True,
    )
    conf_sources = _prefer_sources(
        extracted.logistics.sources if extracted.logistics else [],
        all_sources
    )

    await _add_and_verify(
        evaluator,
        conf_node,
        "location",
        "Conference location identified as Vancouver, Canada",
        "The ICML 2025 conference location (city and country) is Vancouver, Canada.",
        conf_sources
    )

    await _add_and_verify(
        evaluator,
        conf_node,
        "main_conference_dates",
        "Main conference sessions dates identified as July 13–19, 2025",
        "The main conference sessions for ICML 2025 are scheduled for July 13–19, 2025.",
        conf_sources
    )

    # 2) Submission deadlines
    dl_node = evaluator.add_parallel(
        id="submission_deadlines",
        desc="Critical submission deadlines are correctly provided with timezone",
        parent=root,
        critical=True,
    )
    dl_sources = _prefer_sources(
        extracted.submission_deadlines.sources if extracted.submission_deadlines else [],
        all_sources
    )

    await _add_and_verify(
        evaluator,
        dl_node,
        "abstract_deadline",
        "Abstract submission deadline provided as January 23, 2025 in AoE",
        "The abstract submission deadline for ICML 2025 is January 23, 2025 (AoE).",
        dl_sources
    )

    await _add_and_verify(
        evaluator,
        dl_node,
        "full_paper_deadline",
        "Full paper submission deadline provided as January 30, 2025 in AoE",
        "The full paper submission deadline for ICML 2025 is January 30, 2025 (AoE).",
        dl_sources
    )

    await _add_and_verify(
        evaluator,
        dl_node,
        "supplementary_deadline",
        "Supplementary material deadline correctly stated as having no separate deadline (same as full paper: January 30, 2025 AoE)",
        "ICML 2025 has no separate deadline for supplementary material; it is due with the full paper on January 30, 2025 (AoE).",
        dl_sources,
        add_ins="Confirm that supplementary material is due at the full paper deadline and that no separate supplementary deadline exists."
    )

    await _add_and_verify(
        evaluator,
        dl_node,
        "timezone_specified",
        "Timezone for deadlines is explicitly stated as Anywhere on Earth (AoE) (or equivalently UTC−12)",
        "ICML 2025 uses the AoE (Anywhere on Earth, UTC−12) timezone for its submission deadlines.",
        dl_sources
    )

    # 3) Paper format specifications
    fmt_node = evaluator.add_parallel(
        id="paper_format_specifications",
        desc="Paper format specs requested are correctly provided",
        parent=root,
        critical=True,
    )
    fmt_sources = _prefer_sources(
        extracted.paper_format_specifications.sources if extracted.paper_format_specifications else [],
        all_sources
    )

    await _add_and_verify(
        evaluator,
        fmt_node,
        "initial_page_limit",
        "Initial submission maximum page limit for the main paper body is stated as 8 pages, with references/appendices/impact statement not counting toward the limit",
        "For initial submission, the main paper body is limited to 8 pages; references, appendices, and the impact statement do not count toward this limit.",
        fmt_sources
    )

    await _add_and_verify(
        evaluator,
        fmt_node,
        "camera_ready_page_limit",
        "Camera-ready maximum page limit for the main paper body is stated as 9 pages",
        "The camera-ready version may have up to 9 pages for the main paper body.",
        fmt_sources
    )

    await _add_and_verify(
        evaluator,
        fmt_node,
        "initial_file_size_limit",
        "Maximum initial submission file size is stated as 50MB",
        "The maximum file size for the initial submission is 50MB.",
        fmt_sources
    )

    await _add_and_verify(
        evaluator,
        fmt_node,
        "camera_ready_file_size_limit",
        "Maximum camera-ready submission file size is stated as 20MB",
        "The maximum file size for the camera-ready submission is 20MB.",
        fmt_sources
    )

    await _add_and_verify(
        evaluator,
        fmt_node,
        "file_and_typesetting_format",
        "Submission format requirement stated as PDF produced using LaTeX (no other typesetting software supported)",
        "ICML 2025 submissions must be in PDF produced using LaTeX; other typesetting software is not supported.",
        fmt_sources
    )

    # 4) Required components
    req_node = evaluator.add_parallel(
        id="required_components",
        desc="Required LaTeX style and mandatory statements are correctly identified",
        parent=root,
        critical=True,
    )
    req_sources = _prefer_sources(
        extracted.required_components.sources if extracted.required_components else [],
        all_sources
    )

    await _add_and_verify(
        evaluator,
        req_node,
        "latex_style",
        "Required LaTeX style package stated as icml2025 (or icml2025 style files)",
        "The required LaTeX style package for ICML 2025 is icml2025 (ICML 2025 style files).",
        req_sources
    )

    await _add_and_verify(
        evaluator,
        req_node,
        "impact_statement",
        "Impact statement is stated as mandatory and stated as not counting toward the page limit",
        "An Impact Statement is mandatory for ICML 2025 and it does not count toward the page limit.",
        req_sources
    )

    await _add_and_verify(
        evaluator,
        req_node,
        "double_blind_anonymization",
        "Double-blind review requirement is stated, including that anonymization is required",
        "ICML 2025 uses double-blind review and submissions must be anonymized.",
        req_sources
    )

    # 5) Reviewer obligations
    rev_node = evaluator.add_parallel(
        id="reviewer_obligations",
        desc="Reviewer obligations are correctly described for first-time submitters and for authors with 4+ submissions",
        parent=root,
        critical=True,
    )
    rev_sources = _prefer_sources(
        extracted.reviewer_obligations.sources if extracted.reviewer_obligations else [],
        all_sources
    )

    await _add_and_verify(
        evaluator,
        rev_node,
        "at_least_one_reviewer_per_submission",
        "States that at least one author per submission must agree to serve as a reviewer, including the existence of exemptions (e.g., insufficient qualifications or organizing roles)",
        "For ICML 2025, at least one author per submission must agree to serve as a reviewer, with exemptions such as insufficient qualifications or organizing roles.",
        rev_sources
    )

    await _add_and_verify(
        evaluator,
        rev_node,
        "four_or_more_submissions_rule",
        "States that authors with 4 or more submissions must agree to serve as reviewers, including exemptions (e.g., AC/SAC/organizing roles)",
        "Authors with 4 or more submissions to ICML 2025 must agree to serve as reviewers, subject to exemptions such as AC/SAC or organizing roles.",
        rev_sources
    )

    await _add_and_verify(
        evaluator,
        rev_node,
        "first_time_submitter_applicability",
        "Explicitly explains how the reviewer obligation applies to first-time submitters (i.e., no special exception beyond the stated exemptions/qualification criteria)",
        "First-time submitters are subject to the same reviewer obligation; there is no special exception beyond the usual exemptions or qualification criteria.",
        rev_sources
    )

    # 6) Platform requirements
    plat_node = evaluator.add_parallel(
        id="platform_requirements",
        desc="Submission platform and suggested account-creation timing are correctly identified",
        parent=root,
        critical=True,
    )
    plat_sources = _prefer_sources(
        extracted.platform_requirements.sources if extracted.platform_requirements else [],
        all_sources
    )

    await _add_and_verify(
        evaluator,
        plat_node,
        "platform_name",
        "Submission platform is identified as OpenReview",
        "ICML 2025 uses OpenReview as the submission platform.",
        plat_sources
    )

    await _add_and_verify(
        evaluator,
        plat_node,
        "suggested_account_creation_deadline",
        "Suggested deadline for creating an account is identified as January 9, 2025",
        "The suggested deadline for creating an OpenReview account for ICML 2025 is January 9, 2025.",
        plat_sources
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                  #
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_icml2025(),
        template_class=ICML2025Extraction,
        extraction_name="icml2025_extraction",
    )

    evaluator.add_ground_truth(EXPECTED, gt_type="expected_icml2025_requirements")

    await build_and_verify_tree(evaluator, root, extracted)

    return evaluator.get_summary()