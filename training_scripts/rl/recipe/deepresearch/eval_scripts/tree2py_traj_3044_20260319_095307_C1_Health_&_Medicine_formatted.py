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
TASK_ID = "cms_balance_announcement_date"
TASK_DESCRIPTION = (
    "When did the Centers for Medicare & Medicaid Services (CMS) officially announce the BALANCE Model for expanding "
    "access to GLP-1 medications? Please provide the specific date and cite an official government source for your answer."
)
EXPECTED_ANNOUNCEMENT_DATE = "February 20, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BalanceAnnouncementExtraction(BaseModel):
    announcement_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_balance_announcement() -> str:
    return """
    Extract from the provided answer:
    1) announcement_date: the specific calendar date that the answer claims for when CMS officially announced the BALANCE Model.
       - Return it exactly as written in the answer (e.g., "February 20, 2026", "Feb. 20, 2026", "2026-02-20", "2/20/2026").
       - If multiple dates are mentioned, choose the one presented as the announcement date for the CMS BALANCE Model.
       - If no such date is stated, return null.
    2) source_urls: a list of all URLs cited in the answer as sources or references (include every URL explicitly present,
       whether in plain text or markdown links; do not invent any).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_government_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        host = host.lower()
        if not host:
            return False
        # Allow cms.gov and other .gov domains (e.g., hhs.gov, medicare.gov, medicaid.gov)
        if host.endswith(".gov") or host == "gov" or host.endswith("gov"):
            return True
        return False
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root,
    extracted: BalanceAnnouncementExtraction
) -> None:
    # Create the critical parent node for this rubric item
    balance_node = evaluator.add_parallel(
        id="BALANCE_Model_Announcement_Date",
        desc="Verify the official announcement date of the CMS BALANCE Model for GLP-1 medication access",
        parent=root,
        critical=True
    )

    # Leaf 1: Correct_Date — the answer must state Feb 20, 2026 (format variations acceptable)
    correct_date_leaf = evaluator.add_leaf(
        id="Correct_Date",
        desc="The answer states that the CMS BALANCE Model was announced on February 20, 2026",
        parent=balance_node,
        critical=True
    )
    stated_date = extracted.announcement_date or ""
    claim_date_match = (
        f"The two dates refer to the same calendar day: '{stated_date}' and '{EXPECTED_ANNOUNCEMENT_DATE}'. "
        f"Accept reasonable formatting variants like 'Feb. 20, 2026', '2026-02-20', or '2/20/2026'."
    )
    await evaluator.verify(
        claim=claim_date_match,
        node=correct_date_leaf,
        additional_instruction="Treat minor punctuation and abbreviation differences (e.g., 'Feb.' vs 'February') as equivalent."
    )

    # Leaf 2: Official_Source_Referenced — at least one official (government) URL is cited
    official_urls = [u for u in (extracted.source_urls or []) if is_government_url(u)]
    evaluator.add_custom_node(
        result=len(official_urls) > 0,
        id="Official_Source_Referenced",
        desc="The answer cites the official CMS press release or another authoritative government source (e.g., CMS.gov)",
        parent=balance_node,
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

    # Extract the claimed announcement date and cited sources from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_balance_announcement(),
        template_class=BalanceAnnouncementExtraction,
        extraction_name="balance_announcement_extraction"
    )

    # Add ground truth info
    evaluator.add_ground_truth(
        {
            "expected_announcement_date": EXPECTED_ANNOUNCEMENT_DATE,
            "require_official_source": True,
            "acceptable_official_domains_example": ["cms.gov", "hhs.gov", "medicare.gov", "medicaid.gov", "*.gov"]
        },
        gt_type="ground_truth"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()