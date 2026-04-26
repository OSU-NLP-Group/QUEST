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
TASK_ID = "icml_grant_info_2025"
TASK_DESCRIPTION = (
    "As a researcher preparing to submit a paper to the International Conference on Machine Learning (ICML) 2025, "
    "I need to compile comprehensive submission information and research metrics for a grant proposal. Please provide "
    "ICML 2025 submission requirements, MIT arXiv institutional ranking metrics, NeurIPS 2024 conference metrics, and "
    "optional Nature journal impact factors, each with supporting reference URLs from official sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ICMLSubmission(BaseModel):
    page_limit: Optional[str] = None
    page_limit_urls: List[str] = Field(default_factory=list)

    document_format: Optional[str] = None
    document_format_urls: List[str] = Field(default_factory=list)

    anonymization_policy: Optional[str] = None
    anonymization_policy_urls: List[str] = Field(default_factory=list)

    full_paper_deadline_timezone: Optional[str] = None
    full_paper_deadline_urls: List[str] = Field(default_factory=list)

    camera_ready_max_file_size: Optional[str] = None
    camera_ready_max_file_size_urls: List[str] = Field(default_factory=list)


class MITArxivRanking(BaseModel):
    ranking_position: Optional[str] = None
    ranking_urls: List[str] = Field(default_factory=list)

    average_submission_count: Optional[str] = None
    average_submission_urls: List[str] = Field(default_factory=list)


class NeurIPS2024Metrics(BaseModel):
    acceptance_rate: Optional[str] = None
    acceptance_rate_urls: List[str] = Field(default_factory=list)

    total_submissions: Optional[str] = None
    total_submissions_urls: List[str] = Field(default_factory=list)

    accepted_papers: Optional[str] = None
    accepted_papers_urls: List[str] = Field(default_factory=list)


class NatureJournalMetrics(BaseModel):
    impact_factor_2024: Optional[str] = None
    impact_factor_2024_urls: List[str] = Field(default_factory=list)

    five_year_impact_factor: Optional[str] = None
    five_year_impact_factor_urls: List[str] = Field(default_factory=list)


class SubmissionAndMetricsExtraction(BaseModel):
    icml_submission: Optional[ICMLSubmission] = None
    mit_arxiv_ranking: Optional[MITArxivRanking] = None
    neurips_2024_metrics: Optional[NeurIPS2024Metrics] = None
    nature_metrics: Optional[NatureJournalMetrics] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the requested information exactly as presented in the answer. For each item, also extract the reference URLs explicitly mentioned in the answer text. Return null for any missing values.

    Structure your JSON as follows:

    {
      "icml_submission": {
        "page_limit": <string or null>,
        "page_limit_urls": [<url>, ...],
        "document_format": <string or null>,
        "document_format_urls": [<url>, ...],
        "anonymization_policy": <string or null>,
        "anonymization_policy_urls": [<url>, ...],
        "full_paper_deadline_timezone": <string or null>,
        "full_paper_deadline_urls": [<url>, ...],
        "camera_ready_max_file_size": <string or null>,
        "camera_ready_max_file_size_urls": [<url>, ...]
      },
      "mit_arxiv_ranking": {
        "ranking_position": <string or null>,
        "ranking_urls": [<url>, ...],
        "average_submission_count": <string or null>,
        "average_submission_urls": [<url>, ...]
      },
      "neurips_2024_metrics": {
        "acceptance_rate": <string or null>,
        "acceptance_rate_urls": [<url>, ...],
        "total_submissions": <string or null>,
        "total_submissions_urls": [<url>, ...],
        "accepted_papers": <string or null>,
        "accepted_papers_urls": [<url>, ...]
      },
      "nature_metrics": {
        "impact_factor_2024": <string or null>,
        "impact_factor_2024_urls": [<url>, ...],
        "five_year_impact_factor": <string or null>,
        "five_year_impact_factor_urls": [<url>, ...]
      }
    }

    URL extraction rules:
    - Extract only URLs explicitly present in the answer (plain URLs or markdown links).
    - Include complete URLs (with http/https). If missing protocol, prepend http://.
    - If multiple URLs are provided for an item, include all of them.
    - If no URL is provided for an item, return an empty array for that item's URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions for additional instructions                                #
# --------------------------------------------------------------------------- #
def ai_ins_icml(field_label: str) -> str:
    return (
        f"Verify the ICML 2025 {field_label} using only official ICML pages (prefer domains like icml.cc; "
        "author guidelines or call-for-papers pages). The claim should be directly supported by the page content. "
        "Allow reasonable phrasing variants (e.g., 'N pages', 'at most N pages'). For timezone in deadlines, ensure "
        "a timezone is explicitly stated on the page. If the provided URL is not from an official ICML page or does "
        "not contain the requested information, mark as not supported."
    )


def ai_ins_arxiv(field_label: str) -> str:
    return (
        f"Verify MIT's {field_label} strictly against the official arXiv institutional submissions report for 2024 "
        "(based on 2022–2024 averages). Prefer arxiv.org or info.arxiv.org domains. Allow minor formatting variants; "
        "do not accept third-party blogs. If the URL is irrelevant or does not show the requested metric, mark as not supported."
    )


def ai_ins_neurips(field_label: str) -> str:
    return (
        f"Verify NeurIPS 2024 main conference {field_label} using neurips.cc official statistics/news pages. "
        "Ensure it is for the main track (not workshops). Allow minor rounding differences for percentages. "
        "If the URL is not neurips.cc or does not show the metric, mark as not supported."
    )


def ai_ins_nature(field_label: str) -> str:
    return (
        f"Verify Nature journal's {field_label} using official Nature pages (nature.com) or Nature's journal metrics pages. "
        "If the provided URL is not an official Nature site or does not show the metric, mark as not supported."
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_icml_submission(
    evaluator: Evaluator,
    parent_node,
    icml: Optional[ICMLSubmission],
) -> None:
    """Build and verify the ICML 2025 submission requirements subtree."""
    group = evaluator.add_parallel(
        id="icml_2025_submission_requirements",
        desc="ICML 2025 submission requirement details with supporting official ICML conference website sources",
        parent=parent_node,
        critical=True,
    )

    # Page limit
    exists_page_limit = evaluator.add_custom_node(
        result=bool(icml and icml.page_limit and icml.page_limit_urls),
        id="page_limit_value_and_url_provided",
        desc="Page limit value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    page_limit_leaf = evaluator.add_leaf(
        id="page_limit_with_url",
        desc="State the ICML 2025 main-body page limit (excluding references/appendices) AND provide a supporting reference URL from the official ICML 2025 website",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ICML 2025 main-body page limit (excluding references and appendices) is '{icml.page_limit if icml else ''}'.",
        node=page_limit_leaf,
        sources=(icml.page_limit_urls if icml else []),
        additional_instruction=ai_ins_icml("main-body page limit (excluding references/appendices)"),
    )

    # Document format
    exists_doc_fmt = evaluator.add_custom_node(
        result=bool(icml and icml.document_format and icml.document_format_urls),
        id="document_format_value_and_url_provided",
        desc="Document format value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    doc_fmt_leaf = evaluator.add_leaf(
        id="document_format_with_url",
        desc="State the required ICML 2025 submission document format AND provide a supporting reference URL from the official ICML 2025 website",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The required ICML 2025 submission document format is '{icml.document_format if icml else ''}'.",
        node=doc_fmt_leaf,
        sources=(icml.document_format_urls if icml else []),
        additional_instruction=ai_ins_icml("submission document format"),
    )

    # Anonymization policy
    exists_anon = evaluator.add_custom_node(
        result=bool(icml and icml.anonymization_policy and icml.anonymization_policy_urls),
        id="anonymization_policy_value_and_url_provided",
        desc="Anonymization policy value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    anon_leaf = evaluator.add_leaf(
        id="anonymization_policy_with_url",
        desc="State the ICML 2025 peer review anonymization/double-blind policy AND provide a supporting reference URL from the official ICML 2025 website",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ICML 2025 peer review anonymization/double-blind policy is '{icml.anonymization_policy if icml else ''}'.",
        node=anon_leaf,
        sources=(icml.anonymization_policy_urls if icml else []),
        additional_instruction=ai_ins_icml("peer review anonymization/double-blind policy"),
    )

    # Full paper deadline with timezone
    exists_deadline = evaluator.add_custom_node(
        result=bool(icml and icml.full_paper_deadline_timezone and icml.full_paper_deadline_urls),
        id="full_paper_deadline_timezone_value_and_url_provided",
        desc="Full paper deadline with timezone and at least one URL are provided",
        parent=group,
        critical=True,
    )
    deadline_leaf = evaluator.add_leaf(
        id="full_paper_deadline_with_timezone_with_url",
        desc="State the ICML 2025 full paper submission deadline including timezone AND provide a supporting reference URL from the official ICML 2025 website",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ICML 2025 full paper submission deadline (with timezone) is '{icml.full_paper_deadline_timezone if icml else ''}'.",
        node=deadline_leaf,
        sources=(icml.full_paper_deadline_urls if icml else []),
        additional_instruction=ai_ins_icml("full paper submission deadline with timezone"),
    )

    # Camera-ready max file size
    exists_cr = evaluator.add_custom_node(
        result=bool(icml and icml.camera_ready_max_file_size and icml.camera_ready_max_file_size_urls),
        id="camera_ready_max_file_size_value_and_url_provided",
        desc="Camera-ready max file size and at least one URL are provided",
        parent=group,
        critical=True,
    )
    cr_leaf = evaluator.add_leaf(
        id="camera_ready_max_file_size_with_url",
        desc="State the maximum camera-ready PDF file size for ICML 2025 AND provide a supporting reference URL from the official ICML 2025 website",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The maximum camera-ready PDF file size for ICML 2025 is '{icml.camera_ready_max_file_size if icml else ''}'.",
        node=cr_leaf,
        sources=(icml.camera_ready_max_file_size_urls if icml else []),
        additional_instruction=ai_ins_icml("maximum camera-ready PDF file size"),
    )


async def verify_mit_arxiv_ranking(
    evaluator: Evaluator,
    parent_node,
    mit: Optional[MITArxivRanking],
) -> None:
    """Build and verify MIT arXiv institutional ranking subtree."""
    group = evaluator.add_parallel(
        id="mit_arxiv_ranking",
        desc="MIT rank and average submission count in the 2024 arXiv institutional submissions report (2022–2024 average), with supporting arXiv report URL(s)",
        parent=parent_node,
        critical=True,
    )

    # Rank position
    exists_rank = evaluator.add_custom_node(
        result=bool(mit and mit.ranking_position and mit.ranking_urls),
        id="ranking_position_value_and_url_provided",
        desc="Ranking position value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    rank_leaf = evaluator.add_leaf(
        id="ranking_position_with_url",
        desc="Provide MIT's rank position in the 2024 arXiv institutional submissions report (2022–2024 average basis) AND provide a supporting reference URL from the arXiv institutional report",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"MIT's rank position in the 2024 arXiv institutional submissions report (2022–2024 average) is '{mit.ranking_position if mit else ''}'.",
        node=rank_leaf,
        sources=(mit.ranking_urls if mit else []),
        additional_instruction=ai_ins_arxiv("rank position"),
    )

    # Average submission count
    exists_avg = evaluator.add_custom_node(
        result=bool(mit and mit.average_submission_count and mit.average_submission_urls),
        id="average_submission_count_value_and_url_provided",
        desc="Average submission count value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    avg_leaf = evaluator.add_leaf(
        id="average_submission_count_with_url",
        desc="Provide MIT's average paper submission count (2022–2024 average) as reported in the 2024 arXiv institutional submissions report AND provide a supporting reference URL from the arXiv institutional report",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"MIT's average paper submission count (2022–2024 average) in the 2024 arXiv institutional submissions report is '{mit.average_submission_count if mit else ''}'.",
        node=avg_leaf,
        sources=(mit.average_submission_urls if mit else []),
        additional_instruction=ai_ins_arxiv("average paper submission count (2022–2024 average)"),
    )


async def verify_neurips_2024_metrics(
    evaluator: Evaluator,
    parent_node,
    neu: Optional[NeurIPS2024Metrics],
) -> None:
    """Build and verify NeurIPS 2024 metrics subtree."""
    group = evaluator.add_parallel(
        id="neurips_2024_metrics",
        desc="NeurIPS 2024 main conference track acceptance metrics with supporting official NeurIPS source URL(s)",
        parent=parent_node,
        critical=True,
    )

    # Acceptance rate
    exists_rate = evaluator.add_custom_node(
        result=bool(neu and neu.acceptance_rate and neu.acceptance_rate_urls),
        id="acceptance_rate_value_and_url_provided",
        desc="Acceptance rate value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    rate_leaf = evaluator.add_leaf(
        id="acceptance_rate_with_url",
        desc="Provide the NeurIPS 2024 main track acceptance rate (percentage) AND provide a supporting reference URL from an official NeurIPS document/page",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The NeurIPS 2024 main conference track acceptance rate is '{neu.acceptance_rate if neu else ''}'.",
        node=rate_leaf,
        sources=(neu.acceptance_rate_urls if neu else []),
        additional_instruction=ai_ins_neurips("acceptance rate"),
    )

    # Total submissions
    exists_total = evaluator.add_custom_node(
        result=bool(neu and neu.total_submissions and neu.total_submissions_urls),
        id="total_submissions_value_and_url_provided",
        desc="Total submissions value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    total_leaf = evaluator.add_leaf(
        id="total_submissions_with_url",
        desc="Provide the total number of submissions to the NeurIPS 2024 main conference track AND provide a supporting reference URL from an official NeurIPS document/page",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The total number of submissions to the NeurIPS 2024 main conference track is '{neu.total_submissions if neu else ''}'.",
        node=total_leaf,
        sources=(neu.total_submissions_urls if neu else []),
        additional_instruction=ai_ins_neurips("total number of submissions"),
    )

    # Accepted papers
    exists_accepted = evaluator.add_custom_node(
        result=bool(neu and neu.accepted_papers and neu.accepted_papers_urls),
        id="accepted_papers_value_and_url_provided",
        desc="Accepted papers value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    accepted_leaf = evaluator.add_leaf(
        id="accepted_papers_with_url",
        desc="Provide the number of accepted papers for the NeurIPS 2024 main conference track AND provide a supporting reference URL from an official NeurIPS document/page",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The number of accepted papers for the NeurIPS 2024 main conference track is '{neu.accepted_papers if neu else ''}'.",
        node=accepted_leaf,
        sources=(neu.accepted_papers_urls if neu else []),
        additional_instruction=ai_ins_neurips("number of accepted papers"),
    )


async def verify_nature_metrics_optional(
    evaluator: Evaluator,
    parent_node,
    nature: Optional[NatureJournalMetrics],
) -> None:
    """Build and verify the optional Nature journal metrics subtree."""
    group = evaluator.add_parallel(
        id="nature_journal_metrics_optional",
        desc="Optional: Nature journal impact factor metrics with supporting official Nature metrics URL(s)",
        parent=parent_node,
        critical=False,
    )

    # Impact factor 2024
    exists_if = evaluator.add_custom_node(
        result=bool(nature and nature.impact_factor_2024 and nature.impact_factor_2024_urls),
        id="impact_factor_2024_value_and_url_provided",
        desc="2024 impact factor value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    if_leaf = evaluator.add_leaf(
        id="impact_factor_2024_with_url",
        desc="Provide Nature's 2024 journal impact factor AND provide a supporting reference URL from Nature’s official metrics pages",
        parent=group,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Nature journal's 2024 journal impact factor is '{nature.impact_factor_2024 if nature else ''}'.",
        node=if_leaf,
        sources=(nature.impact_factor_2024_urls if nature else []),
        additional_instruction=ai_ins_nature("2024 journal impact factor"),
    )

    # Five-year impact factor
    exists_5y = evaluator.add_custom_node(
        result=bool(nature and nature.five_year_impact_factor and nature.five_year_impact_factor_urls),
        id="five_year_impact_factor_value_and_url_provided",
        desc="5-year impact factor value and at least one URL are provided",
        parent=group,
        critical=True,
    )
    five_leaf = evaluator.add_leaf(
        id="five_year_impact_factor_with_url",
        desc="Provide Nature's 2024 5-year impact factor AND provide a supporting reference URL from Nature’s official metrics pages",
        parent=group,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Nature journal's 2024 5-year impact factor is '{nature.five_year_impact_factor if nature else ''}'.",
        node=five_leaf,
        sources=(nature.five_year_impact_factor_urls if nature else []),
        additional_instruction=ai_ins_nature("5-year impact factor (2024)"),
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for ICML 2025 submission requirements, MIT arXiv ranking,
    NeurIPS 2024 metrics, and optional Nature journal metrics.

    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator with a PARALLEL root to allow independent group checks.
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
    # Note: We intentionally keep the root as non-critical to allow optional items.
    # Critical gating is implemented at the group level per rubric.

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=SubmissionAndMetricsExtraction,
        extraction_name="submission_and_metrics",
    )

    # Build verification subtrees according to rubric
    # ICML 2025 Submission Requirements (Critical Group)
    await verify_icml_submission(
        evaluator=evaluator,
        parent_node=root,
        icml=extracted.icml_submission,
    )

    # MIT arXiv Ranking (Critical Group)
    await verify_mit_arxiv_ranking(
        evaluator=evaluator,
        parent_node=root,
        mit=extracted.mit_arxiv_ranking,
    )

    # NeurIPS 2024 Metrics (Critical Group)
    await verify_neurips_2024_metrics(
        evaluator=evaluator,
        parent_node=root,
        neu=extracted.neurips_2024_metrics,
    )

    # Nature Journal Metrics (Optional, Non-Critical Group)
    await verify_nature_metrics_optional(
        evaluator=evaluator,
        parent_node=root,
        nature=extracted.nature_metrics,
    )

    # Return structured result
    return evaluator.get_summary()