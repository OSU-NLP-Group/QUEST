import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "thanksgiving_chain_2024"
TASK_DESCRIPTION = (
    "Identify a major national grocery store chain that meets all of the following criteria for Thanksgiving Day 2024: "
    "(1) The chain operates stores in multiple U.S. states, "
    "(2) The chain is open on Thanksgiving Day 2024, "
    "(3) The chain closes before 6:00 p.m. on Thanksgiving Day 2024, "
    "(4) The chain's pharmacy department is closed on Thanksgiving Day 2024, even though the main store is open. "
    "Provide the name of the grocery store chain and include URL references that verify each of these criteria."
)


class ChainCriteriaExtraction(BaseModel):
    chain_name: Optional[str] = None
    multi_state_urls: List[str] = Field(default_factory=list)
    open_thanksgiving_urls: List[str] = Field(default_factory=list)
    close_before_6pm_urls: List[str] = Field(default_factory=list)
    pharmacy_closed_urls: List[str] = Field(default_factory=list)


def prompt_extract_chain_and_sources() -> str:
    return """
    From the provided answer, extract the following fields related to a proposed grocery store chain and the cited URLs that verify each criterion for Thanksgiving Day 2024. Only extract information explicitly present in the answer.

    Required JSON fields:
    - chain_name: The exact name of the proposed grocery store chain (string). If the answer proposes multiple chains, choose the primary one the answer is evaluating. If no chain name is provided, return null.
    - multi_state_urls: Array of URL strings that demonstrate the chain is a major grocery store chain operating in multiple U.S. states (e.g., company profile pages stating presence in multiple states, store locator pages indicating stores across states, credible news or industry sources, etc.). Deduplicate and include only valid URLs.
    - open_thanksgiving_urls: Array of URL strings that indicate the chain is open on Thanksgiving Day 2024 (November 28, 2024). Prefer official chain announcements, store hours pages, or credible news sources referring specifically to Thanksgiving 2024.
    - close_before_6pm_urls: Array of URL strings that indicate the chain closes before 6:00 p.m. local time on Thanksgiving Day 2024. Prefer official chain/store hours pages; ensure the date context is Thanksgiving 2024.
    - pharmacy_closed_urls: Array of URL strings that indicate the chain’s pharmacy department is closed on Thanksgiving Day 2024. Prefer official pharmacy hours pages or credible chain announcements. If a source also states main stores are open while pharmacy is closed, include it here; otherwise, extract open/closed sources into their respective fields.

    Rules:
    - Return null for chain_name if not provided explicitly.
    - For each URL field, extract only valid, complete URLs. Do not invent or infer URLs.
    - If the answer provides a single "Sources" list without categorization, allocate URLs into the most appropriate fields based on content.
    - Remove duplicates and obviously invalid URLs.
    """


def _sanitize_urls(urls: List[str], max_count: int = 12) -> List[str]:
    """
    Sanitize a list of URLs: remove empties/whitespace, deduplicate while preserving order,
    and limit to max_count to keep verification efficient.
    """
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not u:
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2.lower().startswith(("http://", "https://")) and u2 not in seen:
            cleaned.append(u2)
            seen.add(u2)
        elif ("." in u2) and (u2 not in seen):
            # If missing protocol, prepend http:// per extractor special rules
            u3 = "http://" + u2
            cleaned.append(u3)
            seen.add(u3)
    return cleaned[:max_count]


async def _build_verification_tree(
    evaluator: Evaluator,
    extraction: ChainCriteriaExtraction,
    root: Any
) -> None:
    """
    Build the verification tree and execute verifications according to the rubric.
    """
    # Top-level critical sequential node to enforce gating and no partial credit when critical children fail
    task_main = evaluator.add_sequential(
        id="Root",
        desc="Identify a major national grocery store chain that meets all listed Thanksgiving Day 2024 criteria and provide URL references verifying each criterion.",
        parent=root,
        critical=True
    )

    # Chain_Name: critical existence of chain name
    chain_present = bool(extraction.chain_name and extraction.chain_name.strip())
    evaluator.add_custom_node(
        result=chain_present,
        id="Chain_Name",
        desc="Provide the name of the grocery store chain being proposed.",
        parent=task_main,
        critical=True
    )

    # Criteria_With_Verification: critical parallel node with 4 required verifications
    criteria_node = evaluator.add_parallel(
        id="Criteria_With_Verification",
        desc="Verify (with URLs) that the proposed chain satisfies each stated criterion for Thanksgiving Day 2024.",
        parent=task_main,
        critical=True
    )

    chain = extraction.chain_name or ""

    # Multi_State_And_Major_Chain
    multi_state_node = evaluator.add_leaf(
        id="Multi_State_And_Major_Chain",
        desc="Evidence (via URL reference(s)) shows the chain is a major grocery store chain operating in multiple U.S. states.",
        parent=criteria_node,
        critical=True
    )
    ms_urls = _sanitize_urls(extraction.multi_state_urls)
    ms_claim = f"{chain} is a major grocery store chain operating in multiple U.S. states."
    if ms_urls:
        await evaluator.verify(
            claim=ms_claim,
            node=multi_state_node,
            sources=ms_urls,
            additional_instruction=(
                "Look for explicit indications that the chain operates in more than one U.S. state "
                "(e.g., 'stores in X states', 'nationwide', store locator listing multiple states). "
                "Treat 'major chain' as widely recognized large-scale operation (e.g., large store count "
                "or national/multi-state presence). If the URLs are irrelevant or do not support these points, mark as not supported."
            )
        )
    else:
        multi_state_node.score = 0.0
        multi_state_node.status = "failed"

    # Open_Thanksgiving_2024
    open_node = evaluator.add_leaf(
        id="Open_Thanksgiving_2024",
        desc="Evidence (via URL reference(s)) shows the chain is open on Thanksgiving Day 2024.",
        parent=criteria_node,
        critical=True
    )
    open_urls = _sanitize_urls(extraction.open_thanksgiving_urls)
    open_claim = f"{chain} stores are open on Thanksgiving Day 2024 (November 28, 2024)."
    if open_urls:
        await evaluator.verify(
            claim=open_claim,
            node=open_node,
            sources=open_urls,
            additional_instruction=(
                "Confirm the page indicates that stores are open specifically on Thanksgiving Day 2024. "
                "Prefer official store-hours pages or credible announcements/news. "
                "If the year is not clearly 2024 or the source is irrelevant, mark as not supported."
            )
        )
    else:
        open_node.score = 0.0
        open_node.status = "failed"

    # Closes_Before_6PM_Thanksgiving_2024
    closes_node = evaluator.add_leaf(
        id="Closes_Before_6PM_Thanksgiving_2024",
        desc="Evidence (via URL reference(s)) shows the chain closes before 6:00 p.m. on Thanksgiving Day 2024.",
        parent=criteria_node,
        critical=True
    )
    close_urls = _sanitize_urls(extraction.close_before_6pm_urls)
    closes_claim = f"On Thanksgiving Day 2024, {chain} stores close before 6:00 p.m. local time."
    if close_urls:
        await evaluator.verify(
            claim=closes_claim,
            node=closes_node,
            sources=close_urls,
            additional_instruction=(
                "Verify an explicit closing time on Thanksgiving Day 2024 that is strictly earlier than 6:00 p.m. "
                "(e.g., 2 p.m., 3 p.m., 5 p.m.). If a source states closing at 6:00 p.m. or later, or the date context "
                "is not Thanksgiving 2024, mark as not supported."
            ),
            extra_prerequisites=[open_node]  # Closing time is meaningful only if the store is open
        )
    else:
        closes_node.score = 0.0
        closes_node.status = "failed"

    # Pharmacy_Closed_Thanksgiving_2024
    pharmacy_node = evaluator.add_leaf(
        id="Pharmacy_Closed_Thanksgiving_2024",
        desc="Evidence (via URL reference(s)) shows the chain’s pharmacy department is closed on Thanksgiving Day 2024. (Together with `Open_Thanksgiving_2024`, this satisfies the constraint that the pharmacy is closed while the main store is open.)",
        parent=criteria_node,
        critical=True
    )
    pharm_urls = _sanitize_urls(extraction.pharmacy_closed_urls)
    pharmacy_claim = f"On Thanksgiving Day 2024, {chain}'s pharmacy department is closed."
    if pharm_urls:
        await evaluator.verify(
            claim=pharmacy_claim,
            node=pharmacy_node,
            sources=pharm_urls,
            additional_instruction=(
                "Look for explicit statements that the pharmacy department is closed on Thanksgiving Day 2024. "
                "Pharmacy hours pages or official announcements are ideal. "
                "This node only checks pharmacy closure; the store being open is verified in the separate node."
            ),
            extra_prerequisites=[open_node]  # Enforce the 'pharmacy closed while store open' combined constraint
        )
    else:
        pharmacy_node.score = 0.0
        pharmacy_node.status = "failed"


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
    Entry point: evaluate an answer against the Thanksgiving Day 2024 grocery chain criteria.
    """
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
        default_model=model
    )

    extraction = await evaluator.extract(
        prompt=prompt_extract_chain_and_sources(),
        template_class=ChainCriteriaExtraction,
        extraction_name="chain_and_sources"
    )

    evaluator.add_custom_info(
        info={
            "chain_name_extracted": extraction.chain_name,
            "counts": {
                "multi_state_urls": len(extraction.multi_state_urls or []),
                "open_thanksgiving_urls": len(extraction.open_thanksgiving_urls or []),
                "close_before_6pm_urls": len(extraction.close_before_6pm_urls or []),
                "pharmacy_closed_urls": len(extraction.pharmacy_closed_urls or [])
            }
        },
        info_type="extraction_summary",
        info_name="chain_extraction_overview"
    )

    await _build_verification_tree(evaluator, extraction, root)

    return evaluator.get_summary()