import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "epic_games_status_page"
TASK_DESCRIPTION = "What is the official URL and the official name of Epic Games' server status page where users can check the operational status of Epic Games services?"

EXPECTED_STATUS_URL = "https://status.epicgames.com/"
EXPECTED_PAGE_NAME = "Epic Games Public Status"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StatusPageExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer.
    """
    status_url: Optional[str] = None
    page_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_status_page() -> str:
    return """
    Extract the following information from the provided answer:

    - status_url: The official Epic Games server status page URL as explicitly stated in the answer. This must be the page where users can check the operational status of Epic Games services (i.e., the status homepage). Extract the URL exactly as shown in the answer (do not invent or infer). If not present, return null.
    - page_name: The official page name as explicitly stated in the answer (e.g., the visible title/name on the status page). Extract the string exactly as written in the answer. If not present, return null.
    - sources: A list of all URL(s) that the answer cites as references or sources supporting the status page information. Include any links on epicgames.com or status.epicgames.com if present. If none are present, return an empty array.

    Important:
    - Do not fabricate fields that the answer does not contain.
    - Only extract URLs that are explicitly included in the answer text (including markdown links).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _is_official_epic_url(url: str) -> bool:
    """
    Determine whether the URL belongs to an official Epic Games domain:
    - epicgames.com
    - any subdomain of epicgames.com (including status.epicgames.com)
    """
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        # Accept exact epicgames.com or any *.epicgames.com
        return host == "epicgames.com" or host.endswith(".epicgames.com")
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: StatusPageExtraction,
) -> None:
    """
    Build the verification tree and perform checks according to the rubric.
    """
    # Parent node mirroring the rubric main node (critical, parallel)
    parent_node = evaluator.add_parallel(
        id="Epic_Games_Status_Page_Information",
        desc="Provide the official Epic Games server status page URL and its official page name, with official sourcing.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Official URL is exactly https://status.epicgames.com/
    url_leaf = evaluator.add_leaf(
        id="Official_URL_Is_Primary_Status_Page",
        desc="The URL is exactly https://status.epicgames.com/ (i.e., the primary official Epic Games status page, not a third-party monitoring site).",
        parent=parent_node,
        critical=True
    )
    # Formulate a precise claim that checks the answer content
    claim_url_exact = (
        f"In the provided answer, the official Epic Games status page URL is stated as exactly '{EXPECTED_STATUS_URL}'. "
        f"Do not accept any variation (missing trailing slash, http instead of https, alternate domains, or third-party sites)."
    )
    await evaluator.verify(
        claim=claim_url_exact,
        node=url_leaf,
        additional_instruction=(
            "Only mark Correct if the answer explicitly gives exactly 'https://status.epicgames.com/' "
            "as the official status page URL. Treat any deviation (e.g., missing trailing slash, 'http', or "
            "a non-epicgames.com domain) as Incorrect."
        )
    )

    # 2) Official page name is exactly 'Epic Games Public Status'
    page_name_leaf = evaluator.add_leaf(
        id="Official_Page_Name",
        desc="The official page name is exactly 'Epic Games Public Status'.",
        parent=parent_node,
        critical=True
    )
    claim_page_name_exact = (
        f"In the provided answer, the official page name is stated as exactly '{EXPECTED_PAGE_NAME}'. "
        f"Require exact string match (case and spacing must match)."
    )
    await evaluator.verify(
        claim=claim_page_name_exact,
        node=page_name_leaf,
        additional_instruction=(
            "Only mark Correct if the answer explicitly gives exactly 'Epic Games Public Status' as the page name. "
            "Do not accept synonyms, partial matches, different capitalization, or extra/missing words."
        )
    )

    # 3) Official sourcing provided (from epicgames.com or status.epicgames.com)
    sources = extracted.sources or []
    has_official_source = any(_is_official_epic_url(u) for u in sources)
    evaluator.add_custom_node(
        result=has_official_source,
        id="Official_Sourcing_Provided",
        desc="Provides citation(s)/reference link(s) from official Epic Games resources or support documentation (e.g., epicgames.com or status.epicgames.com) supporting the URL/name.",
        parent=parent_node,
        critical=True
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
    Evaluate an answer for the Epic Games status page task using the Mind2Web2 framework.
    """
    # Initialize evaluator and root
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

    # Extract structured info from the answer
    extracted: StatusPageExtraction = await evaluator.extract(
        prompt=prompt_extract_status_page(),
        template_class=StatusPageExtraction,
        extraction_name="status_page_info"
    )

    # Add ground truth info (for transparency in summary)
    evaluator.add_ground_truth({
        "expected_status_url": EXPECTED_STATUS_URL,
        "expected_page_name": EXPECTED_PAGE_NAME
    }, gt_type="ground_truth_status_page")

    # Build and verify according to rubric
    await build_and_verify_tree(evaluator, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()