import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bengio_scholar_metrics"
TASK_DESCRIPTION = (
    "Find Yoshua Bengio's current h-index and total citation count from his official Google Scholar profile. "
    "Provide both the h-index value and the total citation count, along with the URL to his verified Google Scholar profile page."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CitationMetrics(BaseModel):
    h_index: Optional[str] = None
    total_citations: Optional[str] = None
    profile_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_metrics() -> str:
    return (
        "Extract the following items exactly as they appear in the answer:\n"
        "1) h_index: Yoshua Bengio's h-index value as stated in the answer.\n"
        "2) total_citations: Yoshua Bengio's total citation count as stated in the answer.\n"
        "3) profile_url: The URL to Yoshua Bengio's Google Scholar profile page.\n"
        "Rules:\n"
        "- Return the numeric values as strings, preserving any commas or formatting (e.g., '200,000').\n"
        "- If multiple metrics are mentioned (e.g., 'All' vs. 'Since 2019'), extract the value the answer claims as the current/overall metric; otherwise extract the single value provided.\n"
        "- For the profile URL, extract the exact URL mentioned in the answer (markdown link or plain text). If missing, return null.\n"
        "- If any field is missing in the answer, return null for that field."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_citation_metrics(
    evaluator: Evaluator,
    parent_node,
    extracted: CitationMetrics,
) -> None:
    """
    Build the verification tree for Yoshua Bengio's Google Scholar citation metrics and run verifications.
    """

    # Top-level node representing the rubric section
    metrics_node = evaluator.add_parallel(
        id="Google_Scholar_Citation_Metrics",
        desc="Retrieve citation metrics from Yoshua Bengio's Google Scholar profile",
        parent=parent_node,
        critical=False,
    )

    # Existence checks (critical siblings to gate all other verifications)
    profile_url_exists = evaluator.add_custom_node(
        result=bool(extracted.profile_url and extracted.profile_url.strip()),
        id="profile_url_provided",
        desc="Profile URL is provided in the answer",
        parent=metrics_node,
        critical=True,
    )

    h_index_exists = evaluator.add_custom_node(
        result=bool(extracted.h_index and extracted.h_index.strip()),
        id="h_index_provided",
        desc="H-index value is provided in the answer",
        parent=metrics_node,
        critical=True,
    )

    total_citations_exists = evaluator.add_custom_node(
        result=bool(extracted.total_citations and extracted.total_citations.strip()),
        id="total_citations_provided",
        desc="Total citation count is provided in the answer",
        parent=metrics_node,
        critical=True,
    )

    # 1) Verify the profile URL itself
    profile_url_node = evaluator.add_leaf(
        id="profile_url",
        desc="The URL to Yoshua Bengio's verified Google Scholar profile page",
        parent=metrics_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This page is Yoshua Bengio's Google Scholar citations profile and is a verified profile "
            "(look for 'Verified email' under his name)."
        ),
        node=profile_url_node,
        sources=extracted.profile_url,
        additional_instruction=(
            "Confirm the page is on scholar.google.com and shows 'Yoshua Bengio' at the top. "
            "Also check for 'Verified email' text indicating a verified profile. "
            "If the URL is not a Google Scholar citations profile or not for Yoshua Bengio, mark as not supported."
        ),
    )

    # 2) Verify the h-index value against the profile page
    h_index_node = evaluator.add_leaf(
        id="h_index",
        desc="The h-index value displayed on Yoshua Bengio's Google Scholar profile",
        parent=metrics_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"On Yoshua Bengio's Google Scholar profile, the h-index (All) is {extracted.h_index}.",
        node=h_index_node,
        sources=extracted.profile_url,
        additional_instruction=(
            "Check the 'h-index' number in the metrics table on the profile page, focusing on the 'All' column. "
            "Allow minor formatting differences (e.g., commas, spaces). "
            "If the page shows a different number for h-index (All), mark as not supported."
        ),
    )

    # 3) Verify the total citation count against the profile page
    total_citations_node = evaluator.add_leaf(
        id="total_citations",
        desc="The total citation count displayed on Yoshua Bengio's Google Scholar profile",
        parent=metrics_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"On Yoshua Bengio's Google Scholar profile, the total citations (All) are {extracted.total_citations}.",
        node=total_citations_node,
        sources=extracted.profile_url,
        additional_instruction=(
            "Check the 'Citations' number in the metrics table on the profile page, focusing on the 'All' column. "
            "Allow minor formatting differences (e.g., commas). "
            "If the page shows a different number for total citations (All), mark as not supported."
        ),
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
    """
    Evaluate an answer for the Yoshua Bengio Google Scholar metrics task.
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

    # Extract metrics from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_metrics(),
        template_class=CitationMetrics,
        extraction_name="citation_metrics_extraction",
    )

    # Build verification tree and run checks
    await verify_citation_metrics(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()