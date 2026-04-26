import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "plos_one_apc_timeline"
TASK_DESCRIPTION = (
    "A researcher is preparing to submit a standard research article to PLOS ONE and needs to understand the publication costs and timeline. "
    "According to PLOS ONE's current policies, what is the article processing charge (APC) for a regular research article, and what is the average time "
    "from submission to the first editorial decision?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class APCExtraction(BaseModel):
    """Extracted APC information for a regular PLOS ONE research article."""
    amount: Optional[str] = None  # e.g., "$1,895" or "1895"
    currency: Optional[str] = None  # e.g., "USD", "$", "US$"
    sources: List[str] = Field(default_factory=list)  # URLs cited for APC


class TimelineExtraction(BaseModel):
    """Extracted editorial timeline information for PLOS ONE."""
    average_time: Optional[str] = None  # e.g., "35" or "30-35"
    units: Optional[str] = None  # e.g., "days", "weeks"
    sources: List[str] = Field(default_factory=list)  # URLs cited for timeline


class PLOSInfoExtraction(BaseModel):
    """Full extraction object combining APC and timeline info."""
    apc: Optional[APCExtraction] = None
    timeline: Optional[TimelineExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plos_info() -> str:
    return (
        "Extract from the answer the current APC and the editorial timeline for a standard (regular) PLOS ONE research article.\n"
        "Return a JSON object with two nested objects: 'apc' and 'timeline'.\n\n"
        "For 'apc':\n"
        "- amount: The APC amount exactly as stated in the answer (keep symbols or formatting such as $ or commas).\n"
        "- currency: The currency string as stated (e.g., 'USD', 'US$', '$'). If not explicitly stated, return null.\n"
        "- sources: An array of all URLs cited in the answer that are specifically used to support the APC information.\n\n"
        "For 'timeline':\n"
        "- average_time: The numeric or textual value representing the typical time to first editorial decision (e.g., '35', '30-35').\n"
        "- units: The units associated with the average_time (e.g., 'days', 'weeks').\n"
        "- sources: An array of all URLs cited in the answer that support the timeline information.\n\n"
        "Rules:\n"
        "1. Extract only what is explicitly mentioned in the answer; do not invent data.\n"
        "2. Include only valid URLs that appear in the answer for the respective pieces of information.\n"
        "3. If any field is missing in the answer, set it to null (for amount/currency/average_time/units) or [] for sources.\n"
        "4. The APC must correspond to a regular research article in PLOS ONE (exclude special article types). Extract the value as presented.\n"
        "5. For timeline, we care about the time from submission to the first editorial decision. If the answer uses a synonymous metric (e.g., 'median'), strictly extract the value and units as stated.\n"
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_apc_information(
    evaluator: Evaluator,
    parent_node,
    extracted: PLOSInfoExtraction,
) -> None:
    """
    Build and verify the APC information subtree:
    - Check that APC amount and currency are provided.
    - Check that at least one source URL is provided.
    - Verify the APC claim against the cited official pages via URLs.
    """
    apc_node = evaluator.add_sequential(
        id="apc_information",
        desc="State the current APC amount (with currency) for a regular PLOS ONE research article (excluding special article types) as specified on PLOS's official fee page, and cite that official page/documentation",
        parent=parent_node,
        critical=True,
    )

    apc = extracted.apc or APCExtraction()

    # Existence check for APC amount and currency
    apc_exists = evaluator.add_custom_node(
        result=bool(apc.amount and apc.amount.strip() and apc.currency and apc.currency.strip()),
        id="apc_value_present",
        desc="APC amount and currency are provided for a regular PLOS ONE research article",
        parent=apc_node,
        critical=True,
    )

    # Existence check for sources
    apc_sources_present = evaluator.add_custom_node(
        result=bool(apc.sources and len(apc.sources) > 0),
        id="apc_sources_provided",
        desc="At least one source URL is provided to support the APC information",
        parent=apc_node,
        critical=True,
    )

    # Verification: APC claim supported by sources (official PLOS page/documentation)
    apc_supported_leaf = evaluator.add_leaf(
        id="apc_supported_by_sources",
        desc="The APC amount and currency are supported by official PLOS page(s) cited in the answer",
        parent=apc_node,
        critical=True,
    )

    apc_claim_text = f"The Article Processing Charge (APC) for a standard PLOS ONE research article is {apc.amount} {apc.currency}."
    await evaluator.verify(
        claim=apc_claim_text,
        node=apc_supported_leaf,
        sources=apc.sources,
        additional_instruction=(
            "Verify the APC on an official PLOS webpage or documentation (e.g., plos.org/journals/plosone or related official PLOS fee pages). "
            "The APC must refer to a standard (regular) research article in PLOS ONE, not special article types. "
            "Allow minor formatting variations (e.g., currency symbol position or punctuation). "
            "If the provided URLs are not official PLOS pages or do not explicitly support the APC, mark as unsupported."
        ),
    )


async def verify_timeline_information(
    evaluator: Evaluator,
    parent_node,
    extracted: PLOSInfoExtraction,
) -> None:
    """
    Build and verify the timeline information subtree:
    - Check that average time and units are provided.
    - Check that at least one source URL is provided.
    - Verify the timeline claim against the cited official pages via URLs.
    """
    timeline_node = evaluator.add_sequential(
        id="timeline_information",
        desc="State the average time from submission to first editorial decision (including units, e.g., days) using the figure provided in PLOS ONE's official editorial/peer review process documentation, and cite that official page/documentation",
        parent=parent_node,
        critical=True,
    )

    timeline = extracted.timeline or TimelineExtraction()

    # Existence check for timeline value and units
    timeline_exists = evaluator.add_custom_node(
        result=bool(timeline.average_time and timeline.average_time.strip() and timeline.units and timeline.units.strip()),
        id="timeline_value_present",
        desc="The average time and units from submission to first editorial decision are provided",
        parent=timeline_node,
        critical=True,
    )

    # Existence check for sources
    timeline_sources_present = evaluator.add_custom_node(
        result=bool(timeline.sources and len(timeline.sources) > 0),
        id="timeline_sources_provided",
        desc="At least one source URL is provided to support the editorial timeline information",
        parent=timeline_node,
        critical=True,
    )

    # Verification: Timeline claim supported by official PLOS page(s)
    timeline_supported_leaf = evaluator.add_leaf(
        id="timeline_supported_by_sources",
        desc="The average time to first editorial decision is supported by official PLOS page(s) cited in the answer",
        parent=timeline_node,
        critical=True,
    )

    timeline_claim_text = (
        f"The typical time from submission to the first editorial decision at PLOS ONE is {timeline.average_time} {timeline.units}."
    )
    await evaluator.verify(
        claim=timeline_claim_text,
        node=timeline_supported_leaf,
        sources=timeline.sources,
        additional_instruction=(
            "Verify the time-to-first-editorial-decision on an official PLOS ONE webpage or editorial/peer review process documentation page. "
            "PLOS ONE may report this metric as 'average' or 'median'; treat the primary reported figure as the typical timeframe metric for first decision. "
            "Focus specifically on 'submission to first editorial decision' (not acceptance time). "
            "If the provided URLs are not official PLOS pages or do not support the stated figure, mark as unsupported."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for PLOS ONE APC and editorial timeline information.
    Builds a verification tree with critical checks for APC and timeline, and verifies
    claims against cited official PLOS webpages.
    """
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

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_plos_info(),
        template_class=PLOSInfoExtraction,
        extraction_name="plos_one_info",
    )

    # Build verification subtrees
    await verify_apc_information(evaluator, root, extracted_info)
    await verify_timeline_information(evaluator, root, extracted_info)

    # Return the standardized evaluation summary
    return evaluator.get_summary()