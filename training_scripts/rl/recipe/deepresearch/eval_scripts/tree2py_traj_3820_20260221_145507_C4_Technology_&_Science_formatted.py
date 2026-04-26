import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_outage_2025"
TASK_DESCRIPTION = (
    "In 2025, several major technology service outages significantly disrupted users worldwide and generated millions "
    "of reports on Downdetector, a real-time outage tracking platform. Identify the single largest service outage of "
    "2025 based on total Downdetector user report count across all affected services, and provide the following "
    "information:\n\n"
    "1. The name of the service or company that experienced the outage\n"
    "2. The specific date when the outage occurred\n"
    "3. The total duration of the outage in hours\n"
    "4. The total number of Downdetector user reports received during this incident\n"
    "5. The primary geographic scope of the outage (e.g., Global, United States, Europe, etc.)\n"
    "6. The technical root cause or primary cause category of the outage\n"
    "7. A reference URL from Downdetector's website analyzing or documenting this outage\n"
    "8. A reference URL from a major news outlet covering this outage\n\n"
    "Your response should be based on publicly available information from Downdetector's official reports, analyses, "
    "or status pages, as well as reputable news sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageInfo(BaseModel):
    """Structured information for the largest 2025 outage extracted from the agent's answer."""
    service_name: Optional[str] = None
    outage_date: Optional[str] = None            # Prefer YYYY-MM-DD but keep string to handle variants
    duration_hours: Optional[str] = None         # Keep as string to allow ranges/approximations
    report_count: Optional[str] = None           # Keep as string for numbers with separators/units
    affected_region: Optional[str] = None
    root_cause: Optional[str] = None
    downdetector_url: Optional[str] = None
    news_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_info() -> str:
    return (
        "Extract the single largest service outage of 2025 described in the answer (by total Downdetector user report "
        "count across all affected services) and provide the following fields:\n"
        "- service_name: Name of the service/company impacted (e.g., 'Meta/Facebook', 'Google', 'Cloudflare').\n"
        "- outage_date: The specific date when the outage occurred (prefer YYYY-MM-DD; if different format is used in the answer, extract exactly that string).\n"
        "- duration_hours: Total duration in hours (allow textual approximations like 'about 3 hours').\n"
        "- report_count: Total Downdetector user report count (as presented in the answer; include separators or wording).\n"
        "- affected_region: Primary geographic scope (e.g., Global, United States, Europe).\n"
        "- root_cause: Technical root cause or primary cause category (e.g., DNS issue, configuration error, software bug, cloud provider outage).\n"
        "- downdetector_url: A direct URL from Downdetector's website analyzing or documenting this outage.\n"
        "- news_url: A URL from a major news outlet covering this outage.\n\n"
        "Rules:\n"
        "1) Extract exactly what the answer states; do not invent missing fields.\n"
        "2) For URLs, extract only valid complete URLs explicitly present in the answer text (plain or markdown link). If missing, return null.\n"
        "3) If multiple outages or URLs are listed, pick the one the answer claims is the largest; otherwise pick the first.\n"
        "4) If any field is absent, return null for that field."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _available_sources(info: OutageInfo) -> List[str]:
    sources: List[str] = []
    if info.downdetector_url and info.downdetector_url.strip():
        sources.append(info.downdetector_url.strip())
    if info.news_url and info.news_url.strip():
        sources.append(info.news_url.strip())
    return sources


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def add_and_verify_sources(
    evaluator: Evaluator,
    parent: VerificationNode,
    info: OutageInfo,
) -> Dict[str, VerificationNode]:
    """
    Create and verify the two source leaves (Downdetector and News).
    Returns a dict of nodes to be used as prerequisites for other field verifications.
    """
    # Downdetector reference leaf
    dd_node = evaluator.add_leaf(
        id="Downdetector_Reference",
        desc="A direct URL to Downdetector's analysis or status page for this outage",
        parent=parent,
        critical=True,
    )
    dd_claim = (
        f"This URL is a page on Downdetector (downdetector.com) that documents or analyzes a 2025 outage "
        f"related to '{info.service_name or ''}', and it meaningfully discusses the incident (date, report counts, charts, or analysis)."
    )
    await evaluator.verify(
        claim=dd_claim,
        node=dd_node,
        sources=info.downdetector_url,
        additional_instruction=(
            "Verify the URL domain is Downdetector (e.g., downdetector.com). The page should analyze or document "
            "the specific outage in 2025 for the referenced service/company. Look for clear outage context (date, "
            "charts, timeline, or report count mentions). If the URL is missing, malformed, unrelated, or lacks "
            "clear outage documentation, mark as not supported."
        ),
    )

    # News reference leaf
    news_node = evaluator.add_leaf(
        id="News_Reference",
        desc="A URL to a news article from a major outlet covering this outage",
        parent=parent,
        critical=True,
    )
    news_claim = (
        f"This URL is a news article from a major outlet covering the outage affecting '{info.service_name or ''}' "
        f"in 2025 and provides relevant coverage (date, extent, impact, or cause)."
    )
    await evaluator.verify(
        claim=news_claim,
        node=news_node,
        sources=info.news_url,
        additional_instruction=(
            "Assess whether the source is a recognized major outlet (e.g., Reuters, AP, Bloomberg, WSJ, NYTimes, CNN, "
            "CNBC, BBC, The Guardian, Washington Post, FT, Forbes, The Verge, Wired, TechCrunch, Engadget, etc.). "
            "It must specifically cover the described outage in 2025. If the URL is missing, from a non-reputable site, "
            "or not clearly about the outage, mark as not supported."
        ),
    )

    return {"downdetector": dd_node, "news": news_node}


async def add_and_verify_fields(
    evaluator: Evaluator,
    parent: VerificationNode,
    info: OutageInfo,
    prereq_nodes: List[VerificationNode],
) -> None:
    """
    Create and verify each required field leaf under the parent node.
    Each verification depends on the provided prerequisite source nodes.
    """
    sources = _available_sources(info)

    # Service Name
    svc_node = evaluator.add_leaf(
        id="Service_Name",
        desc="The name of the service or company that experienced the outage",
        parent=parent,
        critical=True,
    )
    svc_claim = (
        f"The outage documented by the provided sources involved the service/company '{info.service_name or ''}'. "
        "Allow synonyms or brand families (e.g., 'Meta' vs 'Facebook') if clearly referring to the same entity."
    )
    await evaluator.verify(
        claim=svc_claim,
        node=svc_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Confirm the service/company named matches the outage coverage on the provided sources. Accept reasonable "
            "name variants (letter casing, minor spelling, brand vs product). If sources do not clearly identify "
            "this service/company for the outage, mark as not supported."
        ),
        extra_prerequisites=prereq_nodes,
    )

    # Outage Date
    date_node = evaluator.add_leaf(
        id="Outage_Date",
        desc="The date when the outage occurred (YYYY-MM-DD format)",
        parent=parent,
        critical=True,
    )
    date_claim = (
        f"The outage occurred on '{info.outage_date or ''}' (accept equivalent date formats that correspond to the same date in 2025)."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Check the sources for the stated outage date. Allow equivalent formats (e.g., 'Jan 2, 2025' == '2025-01-02'). "
            "If sources give a different date or no clear date, mark as not supported."
        ),
        extra_prerequisites=prereq_nodes,
    )

    # Duration in Hours
    dur_node = evaluator.add_leaf(
        id="Duration_Hours",
        desc="The total duration of the outage measured in hours",
        parent=parent,
        critical=True,
    )
    dur_claim = (
        f"The total duration of the outage was '{info.duration_hours or ''}' hours (allow approximate phrasing like 'about N hours')."
    )
    await evaluator.verify(
        claim=dur_claim,
        node=dur_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Verify the sources mention a duration consistent with the claim. Accept approximate phrasing or ranges if "
            "clearly equivalent. If no duration is given or it contradicts the claim, mark as not supported."
        ),
        extra_prerequisites=prereq_nodes,
    )

    # Downdetector Report Count
    reports_node = evaluator.add_leaf(
        id="Downdetector_Reports",
        desc="The total number of user reports received on Downdetector during the outage",
        parent=parent,
        critical=True,
    )
    rpt_claim = (
        f"The total number of Downdetector user reports during this incident was '{info.report_count or ''}' "
        "(focus on total across all affected services for this outage; allow rounding/approximation if clearly indicated)."
    )
    await evaluator.verify(
        claim=rpt_claim,
        node=reports_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Look for explicit counts or clearly stated totals in the Downdetector page or major news article. "
            "Accept reasonable rounding (e.g., '~2 million' vs '2,000,000'). If only per-service charts are shown without an "
            "explicit total and no other credible source provides the total, mark as not supported."
        ),
        extra_prerequisites=prereq_nodes,
    )

    # Affected Region
    region_node = evaluator.add_leaf(
        id="Affected_Region",
        desc="The primary geographic region or scope affected by the outage (e.g., Global, US, Europe)",
        parent=parent,
        critical=True,
    )
    region_claim = (
        f"The primary geographic scope of the outage was '{info.affected_region or ''}' (e.g., Global, United States, Europe)."
    )
    await evaluator.verify(
        claim=region_claim,
        node=region_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Confirm geographical scope from the sources. If multiple countries/regions are explicitly stated and "
            "the outage clearly affected many regions, 'Global' can be acceptable. If sources contradict the claimed scope, fail."
        ),
        extra_prerequisites=prereq_nodes,
    )

    # Root Cause
    cause_node = evaluator.add_leaf(
        id="Root_Cause",
        desc="The technical root cause or category of the outage (e.g., software issue, DNS problem, configuration error)",
        parent=parent,
        critical=True,
    )
    cause_claim = (
        f"The primary technical cause category of the outage was '{info.root_cause or ''}' "
        "(e.g., DNS issue, configuration error, routing problem, cloud provider outage, software bug)."
    )
    await evaluator.verify(
        claim=cause_claim,
        node=cause_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Check the sources for cause description. Allow classification mapping (e.g., 'BGP routing issue' maps to 'routing problem'). "
            "If the cause is uncertain or not stated, mark as not supported."
        ),
        extra_prerequisites=prereq_nodes,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 2025 largest outage task.

    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root-level parallel aggregation
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

    # Create a dedicated node matching the rubric tree's top-level "Major_Outage_Analysis"
    major_node = evaluator.add_parallel(
        id="Major_Outage_Analysis",
        desc="Identify the largest service outage of 2025 by Downdetector user report count and provide comprehensive documentation",
        parent=root,
        critical=False,  # Non-critical to allow partial scoring if some fields pass
    )

    # Extract structured outage info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_outage_info(),
        template_class=OutageInfo,
        extraction_name="largest_outage_2025",
    )

    # Verify source references first (they act as prerequisites for other fields)
    prereqs = await add_and_verify_sources(evaluator, major_node, extracted)
    prereq_nodes = [prereqs["downdetector"], prereqs["news"]]

    # Verify remaining fields, gated by source reference checks
    await add_and_verify_fields(evaluator, major_node, extracted, prereq_nodes)

    # Return summary
    return evaluator.get_summary()