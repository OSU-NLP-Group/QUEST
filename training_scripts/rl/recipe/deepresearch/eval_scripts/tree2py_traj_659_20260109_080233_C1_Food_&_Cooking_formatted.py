import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fda_safe_storage_temps"
TASK_DESCRIPTION = (
    "According to the FDA, what are the recommended safe storage temperatures for a home refrigerator and freezer? "
    "Provide the specific temperature thresholds and include a link to an official FDA webpage that confirms this information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SourceListExtraction(BaseModel):
    """
    Extract all URLs explicitly mentioned in the answer that are intended as citations or references.
    """
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_sources() -> str:
    return """
    Extract all URLs that are explicitly present in the answer and are used as citations, references, or source links
    for the temperature information. Return them in an array called 'sources'. Include:
    - Plain URLs (e.g., https://www.fda.gov/...)
    - Markdown links (extract the actual URL portion)
    - Any other explicit URL strings

    Do not invent URLs. If the answer contains no URLs, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_fda_url(url: str) -> bool:
    """
    Determine whether a given URL belongs to an official FDA domain.

    Criteria:
    - The URL must parse successfully.
    - The hostname must end with 'fda.gov' (e.g., www.fda.gov, www.cfsan.fda.gov, etc.)
    """
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        # If protocol is missing, urlparse may put host in path; try to handle simple cases
        if not host and "://" not in url:
            parsed = urlparse("http://" + url.strip())
            host = (parsed.netloc or "").lower()
        return host.endswith("fda.gov")
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    sources_extraction: SourceListExtraction,
) -> None:
    """
    Build the verification tree per rubric and run verifications.
    """
    # Root-level node in our tree (non-critical root already created by evaluator.initialize)
    # Create a critical parallel node to mirror rubric "FDA_Temperature_Requirements"
    req_node = evaluator.add_parallel(
        id="FDA_Temperature_Requirements",
        desc="Provide FDA-recommended safe storage temperatures for home refrigerators and freezers",
        parent=evaluator.root,
        critical=True
    )

    # 1) Refrigerator requirement: "The refrigerator temperature stated is 40°F or below"
    fridge_leaf = evaluator.add_leaf(
        id="Refrigerator_Temperature",
        desc="The answer states the recommended home refrigerator temperature is 40°F (4°C) or below",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the recommended home refrigerator temperature is 40°F (4°C) or below.",
        node=fridge_leaf,
        additional_instruction=(
            "Judge based on the provided answer text. Accept equivalent phrasings such as '40°F or below', "
            "'40 °F or lower', '≤ 40°F', or Celsius equivalents like '4°C' for the refrigerator. "
            "If the answer is missing a refrigerator threshold, mark incorrect."
        ),
    )

    # 2) Freezer requirement: "The freezer temperature stated is 0°F or below"
    freezer_leaf = evaluator.add_leaf(
        id="Freezer_Temperature",
        desc="The answer states the recommended home freezer temperature is 0°F (-18°C) or below",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the recommended home freezer temperature is 0°F (-18°C) or below.",
        node=freezer_leaf,
        additional_instruction=(
            "Judge based on the provided answer text. Accept equivalent phrasings such as '0°F or below', "
            "'0 °F or lower', '≤ 0°F', or Celsius equivalents like '-18°C' for the freezer. "
            "If the answer is missing a freezer threshold, mark incorrect."
        ),
    )

    # 3) FDA Source: We break into two concrete checks for clarity:
    #    (a) At least one FDA URL provided (existence & domain)
    #    (b) At least one FDA URL supports both thresholds
    all_urls = sources_extraction.sources or []
    fda_urls = [u for u in all_urls if is_fda_url(u)]

    # 3a) Existence of FDA URL (critical)
    fda_url_exists = evaluator.add_custom_node(
        result=len(fda_urls) > 0,
        id="FDA_URL_Provided",
        desc="At least one official FDA (fda.gov) URL is provided in the answer",
        parent=req_node,
        critical=True
    )

    # 3b) The FDA page supports both temperatures (critical)
    fda_support_leaf = evaluator.add_leaf(
        id="FDA_Source_Reference",
        desc="A valid FDA webpage URL is provided that supports the temperature information",
        parent=req_node,
        critical=True
    )
    # Note: Because FDA_URL_Provided is a critical sibling under the same parent, evaluator.verify()
    # will automatically treat it as a prerequisite and skip this verification if that node failed.
    await evaluator.verify(
        claim=(
            "This is an official FDA webpage and it explicitly states the recommended home storage temperatures: "
            "Refrigerator at 40°F (4°C) or below, and Freezer at 0°F (-18°C) or below."
        ),
        node=fda_support_leaf,
        sources=fda_urls,  # Only pass FDA URLs
        additional_instruction=(
            "Treat the page as 'official FDA' if the URL domain ends with fda.gov (the URL is provided). "
            "Look for language that clearly confirms both thresholds, even if phrased as 'at or below', "
            "'or less', 'keep at 40°F or lower', etc. Accept Celsius equivalents (4°C for fridge, -18°C for freezer). "
            "If the page is irrelevant, inaccessible, not FDA, or does not confirm both thresholds, mark as not supported."
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
    Evaluate an answer for FDA recommended safe storage temperatures.
    """
    # Initialize evaluator with a parallel root as rubric suggests independent checks
    evaluator = Evaluator()
    evaluator.initialize(
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

    # 1) Extract all URLs from the answer
    sources_extraction = await evaluator.extract(
        prompt=prompt_extract_sources(),
        template_class=SourceListExtraction,
        extraction_name="extracted_sources",
    )

    # Record expected ground truth for transparency
    evaluator.add_ground_truth(
        {
            "expected_refrigerator": "40°F (4°C) or below",
            "expected_freezer": "0°F (-18°C) or below",
            "source_requirement": "At least one official FDA (fda.gov) webpage confirming both thresholds",
        },
        gt_type="ground_truth"
    )

    # Optionally record filtered FDA URLs for debugging
    evaluator.add_custom_info(
        info={
            "all_extracted_urls": sources_extraction.sources,
            "filtered_fda_urls": [u for u in (sources_extraction.sources or []) if is_fda_url(u)],
        },
        info_type="debug",
        info_name="url_debug_info"
    )

    # 2) Build verification tree and run checks
    await build_and_verify_tree(evaluator, sources_extraction)

    # 3) Return evaluation summary
    return evaluator.get_summary()