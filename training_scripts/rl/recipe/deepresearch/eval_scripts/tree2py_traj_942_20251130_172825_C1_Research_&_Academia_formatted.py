import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "usd_notebooklm_adoption_2025"
TASK_DESCRIPTION = (
    "An academic researcher is preparing a timeline documenting the adoption of AI research tools by U.S. higher education institutions in 2025. "
    "For their study, they need to determine: When did the University of San Diego officially announce the adoption of Google NotebookLM for campus-wide use? "
    "Provide the specific date and cite an official source from the University of San Diego."
)

EXPECTED_ANNOUNCEMENT_DATE = "August 13, 2025"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class USDAdoptionExtraction(BaseModel):
    """
    Extracted info from the agent's answer related to USD's NotebookLM adoption.
    """
    announcement_date: Optional[str] = None
    all_urls: List[str] = Field(default_factory=list)
    usd_urls: List[str] = Field(default_factory=list)
    campus_wide_statement: Optional[str] = None
    effective_same_day_statement: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_usd_notebooklm() -> str:
    return """
    Extract the specific information stated in the answer about the University of San Diego's adoption of Google NotebookLM.

    You must extract exactly and only what the answer text explicitly states.

    Return a JSON object with the following fields:
    - announcement_date: The specific date the answer claims USD officially announced adopting Google NotebookLM for campus-wide use. Keep the original phrasing if present (e.g., "August 13, 2025" or "Aug. 13, 2025"). If multiple dates appear, choose the one explicitly tied to USD's official announcement of NotebookLM campus-wide adoption. If none, return null.
    - all_urls: Array of all URLs explicitly mentioned in the answer (include every URL).
    - usd_urls: Array of URLs that are hosted on the official University of San Diego domain (i.e., URLs whose domain includes "sandiego.edu"). This should be a subset of all_urls. If none, return an empty array.
    - campus_wide_statement: The exact phrase or a concise paraphrase from the answer indicating that NotebookLM is adopted campus-wide and available to all students, faculty, and staff. If not present, return null.
    - effective_same_day_statement: The exact phrase or a concise paraphrase from the answer stating that the adoption/availability was immediate or effective the same day as the announcement (e.g., "effective immediately", "available today"). If not present, return null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pick_usd_news_url(urls: List[str]) -> Optional[str]:
    """
    From a list of USD URLs (sandiego.edu domain), pick one most likely to be an official news/announcement page.
    Preference order: URLs containing '/news' or a known news section.
    Fallback: first USD URL if available.
    """
    if not urls:
        return None

    prioritized_keywords = [
        "/news",
        "news.sandiego.edu",
        "insideusd",
        "/its/news",
        "toreronetwork.sandiego.edu/news",
        "torerotimes.sandiego.edu",
        "www.sandiego.edu/news",
    ]

    def is_news_like(u: str) -> bool:
        low = u.lower()
        return any(k in low for k in prioritized_keywords)

    # Prefer news-like URLs
    news_like = [u for u in urls if is_news_like(u)]
    if news_like:
        return news_like[0]

    # Otherwise, return the first USD URL
    return urls[0]


def is_sandiego_domain(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return "sandiego.edu" in netloc
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_usd_adoption(
    evaluator: Evaluator,
    parent_node,
    extracted: USDAdoptionExtraction,
) -> None:
    """
    Build the verification subtree for USD NotebookLM adoption and run checks.
    """
    # Create a critical parallel parent node for all checks (as per rubric)
    usd_node = evaluator.add_parallel(
        id="USD_NotebookLM_Adoption_Verification",
        desc="Verify the date and official USD announcement details for the adoption of Google NotebookLM for campus-wide use.",
        parent=parent_node,
        critical=True,
    )

    # 1) Official_USD_Source_Cited (critical)
    has_usd_source = any(is_sandiego_domain(u) for u in extracted.usd_urls)
    official_source_node = evaluator.add_custom_node(
        result=has_usd_source,
        id="Official_USD_Source_Cited",
        desc="Provides a citation (e.g., URL) to an official University of San Diego source supporting the claim.",
        parent=usd_node,
        critical=True
    )

    # 2) Source_Is_USD_News_On_Sandiego_Domain (critical)
    usd_news_url = pick_usd_news_url(extracted.usd_urls)
    source_is_usd_news_node = evaluator.add_leaf(
        id="Source_Is_USD_News_On_Sandiego_Domain",
        desc="The cited official source is specifically on USD’s official news website/section and is hosted on the sandiego.edu domain.",
        parent=usd_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is an official University of San Diego news or announcement page (USD News Center, Inside USD, or another USD 'news' section) and is hosted on the sandiego.edu domain.",
        node=source_is_usd_news_node,
        sources=usd_news_url,  # Prefer a single representative news URL
        additional_instruction=(
            "Accept variants like 'USD News Center', 'Inside USD', or a USD department news page under sandiego.edu. "
            "Confirm both: (1) the domain contains sandiego.edu and (2) the page functions as a USD news/announcement article."
        ),
        extra_prerequisites=[official_source_node],
    )

    # 3) Announcement_Date (critical) — verify the specific date via the official USD source(s)
    announcement_date_node = evaluator.add_leaf(
        id="Announcement_Date",
        desc=f"States the official announcement date as {EXPECTED_ANNOUNCEMENT_DATE}.",
        parent=usd_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The official USD page indicates that the announcement/adoption date for Google NotebookLM campus-wide use is {EXPECTED_ANNOUNCEMENT_DATE}."
        ),
        node=announcement_date_node,
        sources=extracted.usd_urls,
        additional_instruction=(
            "Look for a clearly marked 'Published', 'Posted', or article date near the title or byline. "
            "Accept equivalent formats like 'Aug. 13, 2025' as matching August 13, 2025."
        ),
        extra_prerequisites=[official_source_node],
    )

    # 4) Campus_Wide_Availability (critical) — verify campus-wide availability via official source(s)
    campus_wide_node = evaluator.add_leaf(
        id="Campus_Wide_Availability",
        desc="Indicates NotebookLM was adopted for campus-wide use and available to all students, faculty, and staff.",
        parent=usd_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The official USD page explicitly states that Google NotebookLM is adopted campus-wide and is available to all students, faculty, and staff (i.e., the entire campus community)."
        ),
        node=campus_wide_node,
        sources=extracted.usd_urls,
        additional_instruction=(
            "Accept close synonyms such as 'available to the entire campus community', 'all students, faculty & staff', or 'campus-wide availability'. "
            "Reject if the page limits access to only a subset."
        ),
        extra_prerequisites=[official_source_node],
    )

    # 5) Effective_Same_Day (critical) — verify immediacy via official source(s)
    effective_same_day_node = evaluator.add_leaf(
        id="Effective_Same_Day",
        desc="Indicates the adoption was immediate/effective the same day as the announcement.",
        parent=usd_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The official USD page indicates that campus-wide adoption/availability of Google NotebookLM is effective immediately or the same day as the announcement (i.e., {EXPECTED_ANNOUNCEMENT_DATE})."
        ),
        node=effective_same_day_node,
        sources=extracted.usd_urls,
        additional_instruction=(
            "Look for wording such as 'effective immediately', 'available today', 'available now', or a sentence clearly indicating same-day effect."
        ),
        extra_prerequisites=[official_source_node],
    )

    # Record some helpful custom info for debugging/reporting
    evaluator.add_custom_info(
        info={
            "extracted_announcement_date": extracted.announcement_date,
            "all_urls": extracted.all_urls,
            "usd_urls": extracted.usd_urls,
            "preferred_usd_news_url": usd_news_url,
            "expected_date": EXPECTED_ANNOUNCEMENT_DATE
        },
        info_type="extraction_debug",
        info_name="extraction_debug_info"
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
    Evaluate an answer for the USD NotebookLM 2025 adoption announcement task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator; actual critical grouping done under child node
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_usd_notebooklm(),
        template_class=USDAdoptionExtraction,
        extraction_name="usd_notebooklm_extraction"
    )

    # Optional ground truth info for reference
    evaluator.add_ground_truth(
        {
            "expected_announcement_date": EXPECTED_ANNOUNCEMENT_DATE,
            "required_domain": "sandiego.edu",
            "required_sections_hint": ["USD News Center", "Inside USD", "official USD news section"]
        },
        gt_type="expected_facts"
    )

    # Build verification sub-tree and run checks
    await build_and_verify_usd_adoption(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()