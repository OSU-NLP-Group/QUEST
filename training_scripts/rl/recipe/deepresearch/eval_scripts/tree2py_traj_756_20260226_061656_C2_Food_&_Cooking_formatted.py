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
TASK_ID = "thanksgiving_2025_latest_grocery"
TASK_DESCRIPTION = (
    "On Thanksgiving Day 2025, which national grocery store chain stays open the latest, "
    "and what time does it close? Provide the specific closing time and include a reference "
    "URL that confirms this information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LatestClosingExtraction(BaseModel):
    """
    Extract the single identified national grocery chain that the answer claims stays
    open the latest on Thanksgiving Day 2025, along with its closing time and reference URLs.
    """
    store_name: Optional[str] = None
    closing_time: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_latest_closing() -> str:
    return """
    Your goal is to extract exactly one national grocery store chain that the answer claims
    stays open the latest on Thanksgiving Day 2025, the specific closing time stated, and
    all reference URLs provided to support this claim.

    Extract the following fields:
    - store_name: The name of the national grocery store chain that the answer identifies as staying open the latest on Thanksgiving Day 2025.
    - closing_time: The specific closing time stated for Thanksgiving Day 2025 (keep the exact text such as '10 PM', '11:00 p.m. local time', etc.). Do not normalize; copy verbatim from the answer.
    - reference_urls: A list of all URLs (including markdown links) cited to support the store hours and/or the 'latest' comparison. Extract actual URLs only.

    Special rules:
    - If multiple stores are mentioned, choose the one the answer explicitly identifies as the latest. If ambiguity remains, choose the first store that is explicitly presented as the final answer.
    - If the answer provides no URL(s), return an empty list for reference_urls.
    - Only extract URLs that are explicitly present in the answer text. If missing a protocol, prepend 'http://'.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_urls(urls: List[str]) -> List[str]:
    """Keep only plausible HTTP(S) URLs, deduplicate while preserving order."""
    seen = set()
    cleaned: List[str] = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Ensure protocol
        if not (u.startswith("http://") or u.startswith("https://")):
            # Very lenient fix as per special URL rules
            u = "http://" + u
        # Basic sanity check
        if (u.startswith("http://") or u.startswith("https://")) and u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    root,
    extracted: LatestClosingExtraction,
) -> None:
    """
    Build the rubric verification tree and perform checks according to the provided JSON rubric.
    """

    # Top-level task node (critical, sequential)
    task_node = evaluator.add_sequential(
        id="Thanksgiving_Shopping_Task",
        desc="Identify which national grocery store chain stays open the latest on Thanksgiving Day 2025 and provide its closing time with supporting reference",
        parent=root,
        critical=True
    )

    # Child node for identification and accuracy (critical, parallel)
    identify_node = evaluator.add_parallel(
        id="Latest_Closing_Store_Identification",
        desc="Correctly identifies the grocery store with the latest closing time on Thanksgiving 2025 and provides accurate information",
        parent=task_node,
        critical=True
    )

    # Prepare extracted values
    store_name = (extracted.store_name or "").strip()
    closing_time = (extracted.closing_time or "").strip()
    urls = normalize_urls(extracted.reference_urls)

    # 1) Reference URL existence/validity (critical)
    #    This node ensures at least one plausible URL is provided.
    ref_ok = len(urls) > 0
    evaluator.add_custom_node(
        result=ref_ok,
        id="Reference_URL",
        desc="Provides a valid reference URL supporting the store hours information",
        parent=identify_node,
        critical=True
    )

    # 2) Store open status on Thanksgiving Day 2025 (critical)
    open_status_node = evaluator.add_leaf(
        id="Store_Open_Status",
        desc="The identified store is confirmed to be open on Thanksgiving Day 2025",
        parent=identify_node,
        critical=True
    )

    if not (store_name and urls):
        # Missing essential info or evidence -> fail
        open_status_node.score = 0.0
        open_status_node.status = "failed"
    else:
        open_claim = (
            f"The referenced page(s) explicitly indicate that the national grocery store chain "
            f"'{store_name}' is open on Thanksgiving Day 2025 (Thursday, November 27, 2025)."
        )
        await evaluator.verify(
            claim=open_claim,
            node=open_status_node,
            sources=urls,
            additional_instruction=(
                "Confirm that the page mentions Thanksgiving Day 2025 specifically. "
                "If the page refers only to other years (e.g., 2024 or 2023) or does not clearly indicate being open, "
                "treat this as not supported. If it says most stores or select locations are open with special hours, "
                "that's acceptable as 'open'."
            ),
        )

    # 3) Latest closing time correctness and 'latest among national chains' (critical)
    latest_time_node = evaluator.add_leaf(
        id="Latest_Closing_Time",
        desc="The provided closing time is correct for the identified store and is the latest closing time among all national grocery chains open on Thanksgiving 2025",
        parent=identify_node,
        critical=True
    )

    if not (store_name and closing_time and urls):
        latest_time_node.score = 0.0
        latest_time_node.status = "failed"
    else:
        latest_claim = (
            f"On Thanksgiving Day 2025, the chain '{store_name}' closes at '{closing_time}', "
            f"and this is the latest closing time among national U.S. grocery store chains that are open that day."
        )
        await evaluator.verify(
            claim=latest_claim,
            node=latest_time_node,
            sources=urls,
            additional_instruction=(
                "Use the provided source URLs to validate BOTH parts: "
                "(1) that the chain’s Thanksgiving 2025 closing time matches the stated time, and "
                "(2) that this closing time is latest among national grocery chains open that day. "
                "Pages that compare multiple chains’ Thanksgiving hours or explicitly state that a chain stays open later "
                "than others can support the 'latest' claim. If the sources only show a single chain’s hours without "
                "comparing to others, the 'latest among national chains' claim is not supported."
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
    Evaluate an answer for the Thanksgiving 2025 latest-closing grocery chain task.
    """
    # Initialize evaluator with sequential root to mirror rubric
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_latest_closing(),
        template_class=LatestClosingExtraction,
        extraction_name="latest_closing_extraction",
    )

    # Optionally record a quick summary of what we extracted
    evaluator.add_custom_info(
        info={
            "store_name": extracted.store_name,
            "closing_time": extracted.closing_time,
            "reference_urls_count": len(extracted.reference_urls or []),
        },
        info_type="extraction_summary",
    )

    # Build tree and verify according to rubric
    await build_and_verify(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()