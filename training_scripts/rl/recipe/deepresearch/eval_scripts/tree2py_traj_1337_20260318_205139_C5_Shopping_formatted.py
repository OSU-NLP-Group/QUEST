import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_hours_2025_retail"
TASK_DESCRIPTION = """
Identify at least 4 major U.S. retail store chains that remain open until at least 6 p.m. on Christmas Eve 2025. For each chain, provide the following information: 
(1) The specific closing time on Christmas Eve 2025, 
(2) Their operational status on Christmas Day 2025 (open or closed), 
(3) Their operating hours on New Year's Day 2026, and 
(4) An official reference URL that verifies these holiday hours.
"""

XMAS_EVE_STR = "Christmas Eve 2025 (December 24, 2025)"
XMAS_DAY_STR = "Christmas Day 2025 (December 25, 2025)"
NYD_STR = "New Year's Day 2026 (January 1, 2026)"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChainInfo(BaseModel):
    chain_name: Optional[str] = None
    christmas_eve_2025_closing_time: Optional[str] = None
    christmas_day_2025_status: Optional[str] = None
    new_years_day_2026_hours: Optional[str] = None
    # Official reference URLs that (claim to) verify holiday hours/status
    official_urls: List[str] = Field(default_factory=list)
    # Optional homepage or corporate site URL for the brand (if present in answer)
    homepage_url: Optional[str] = None
    # Any extra URLs mentioned for this chain (e.g., press/news pages) if present in the answer
    extra_support_urls: List[str] = Field(default_factory=list)


class ChainsExtraction(BaseModel):
    chains: List[ChainInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chains() -> str:
    return f"""
Extract all candidate retail chains and their holiday-hour details exactly as presented in the answer text. 
Return a JSON object with a "chains" array, where each element has:

- chain_name: string | null
- christmas_eve_2025_closing_time: string | null
  (e.g., "6 PM", "7:00 p.m.", "8pm", "varies by location", "open late until 10 PM")
- christmas_day_2025_status: string | null
  (e.g., "closed", "open", "open limited hours", "varies")
- new_years_day_2026_hours: string | null
  (e.g., "regular hours", "9am–6pm", "reduced hours", "varies")
- official_urls: array of strings (URLs) — Only include URLs that are explicitly cited in the answer as references for these holiday hours/status. If none, return [].
- homepage_url: string | null — If the answer explicitly includes an official homepage/corporate domain for the chain, include it; otherwise null.
- extra_support_urls: array of strings (URLs) — Any additional URLs mentioned for this chain (e.g., newsrooms, store locators) as present in the answer. If none, return [].

Rules:
- Do not infer or invent anything. Use only information explicitly present in the provided answer.
- Keep all times/hours as free-form strings exactly as written in the answer (do not normalize).
- For URLs, extract the real URL targets, including those embedded in markdown links. Only include valid URLs.
- If the answer includes more than 4 chains, extract them all; downstream evaluation will pick the first 4.
- If the answer includes fewer than 4 chains, extract what is present.

The task context dates are:
- {XMAS_EVE_STR}
- {XMAS_DAY_STR}
- {NYD_STR}
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_k_or_pad(items: List[ChainInfo], k: int) -> List[ChainInfo]:
    """Take first k items; if fewer, pad with empty ChainInfo to reach k."""
    result = list(items[:k])
    while len(result) < k:
        result.append(ChainInfo())
    return result


def _combine_sources(chain: ChainInfo) -> List[str]:
    """Combine all potentially useful URLs for generic verifications."""
    sources: List[str] = []
    sources.extend(chain.official_urls or [])
    sources.extend(chain.extra_support_urls or [])
    if chain.homepage_url:
        sources.append(chain.homepage_url)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for u in sources:
        if u and u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Per-candidate verification                                                  #
# --------------------------------------------------------------------------- #
async def verify_candidate_chain(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    chain: ChainInfo,
    idx: int,
) -> VerificationNode:
    """
    Build verification for one candidate chain (#idx+1).
    Returns the candidate chain parent node.
    """
    display_idx = idx + 1
    candidate_node = evaluator.add_parallel(
        id=f"candidate_chain_{display_idx}",
        desc=f"Candidate retail chain #{display_idx} with provided holiday-hour information",
        parent=parent_node,
        critical=False,  # Per rubric: candidate blocks are non-critical; gating handled by their critical leaves
    )

    name = chain.chain_name or f"Chain #{display_idx}"
    all_sources = _combine_sources(chain)
    official_sources = chain.official_urls or []

    # 1) Major U.S. retail chain
    major_leaf = evaluator.add_leaf(
        id=f"candidate_chain_{display_idx}_is_major_us_chain",
        desc="Chain is identified and is a major U.S. retail store chain",
        parent=candidate_node,
        critical=True,
    )
    major_claim = f"The chain '{name}' is a major U.S. retail store chain."
    await evaluator.verify(
        claim=major_claim,
        node=major_leaf,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Use the provided webpages if available to judge that this is a prominent/large chain in the U.S. "
            "Official brand pages, corporate sites, or widely-recognized overviews suffice. "
            "If no URLs are available, use only the provided answer context."
        ),
    )

    # 2) Christmas Eve 2025 closing time is provided and at least 6 p.m.
    ce_time = chain.christmas_eve_2025_closing_time or "<missing>"
    ce_leaf = evaluator.add_leaf(
        id=f"candidate_chain_{display_idx}_christmas_eve_closing",
        desc="Christmas Eve 2025 closing time is provided and is at least 6 p.m.",
        parent=candidate_node,
        critical=True,
    )
    ce_claim = (
        f"For '{name}', the stated closing time on {XMAS_EVE_STR} is '{ce_time}', and this time is at or after 6:00 PM local time."
    )
    await evaluator.verify(
        claim=ce_claim,
        node=ce_leaf,
        sources=official_sources if official_sources else _combine_sources(chain),
        additional_instruction=(
            "Determine whether the page explicitly or clearly implies that stores are open until 6 PM or later on 12/24/2025. "
            "Accept times like '6 PM', '7 PM', '8 PM', '10 PM', or 'open late' that are ≥ 6 PM. "
            "If the page suggests earlier closing (e.g., 5 PM) or provides no clear closing time, judge as NOT supported. "
            "If the extracted string is '<missing>' or only says 'varies by location' without a specific or minimum time ≥ 6 PM, judge as NOT supported."
        ),
    )

    # 3) Christmas Day 2025 operational status provided
    cd_status = chain.christmas_day_2025_status or "<missing>"
    cd_leaf = evaluator.add_leaf(
        id=f"candidate_chain_{display_idx}_christmas_day_status",
        desc="Christmas Day 2025 operational status (open or closed) is provided",
        parent=candidate_node,
        critical=True,
    )
    cd_claim = f"On {XMAS_DAY_STR}, '{name}' stores are {cd_status}."
    await evaluator.verify(
        claim=cd_claim,
        node=cd_leaf,
        sources=official_sources if official_sources else _combine_sources(chain),
        additional_instruction=(
            "Verify that the page states whether the chain is OPEN or CLOSED on 12/25/2025. "
            "Accept clear equivalents like 'closed all day', 'stores are closed', or 'open with limited hours' (if 'open' is stated). "
            "If the extracted string is '<missing>' or the page does not specify Christmas Day status, judge as NOT supported."
        ),
    )

    # 4) New Year's Day 2026 operating hours provided
    nyd_hours = chain.new_years_day_2026_hours or "<missing>"
    nyd_leaf = evaluator.add_leaf(
        id=f"candidate_chain_{display_idx}_new_years_day_hours",
        desc="New Year's Day 2026 operating hours are provided",
        parent=candidate_node,
        critical=True,
    )
    nyd_claim = f"On {NYD_STR}, '{name}' stores operate the following hours: '{nyd_hours}'."
    await evaluator.verify(
        claim=nyd_claim,
        node=nyd_leaf,
        sources=official_sources if official_sources else _combine_sources(chain),
        additional_instruction=(
            "Confirm that the page mentions hours for 1/1/2026 (e.g., 'regular hours', 'reduced hours', or a time range). "
            "If only generic holiday language is present without explicit New Year's Day hours, judge as NOT supported. "
            "If the extracted string is '<missing>', judge as NOT supported."
        ),
    )

    # 5) At least one official reference URL is provided that verifies the stated holiday hours/status
    ref_leaf = evaluator.add_leaf(
        id=f"candidate_chain_{display_idx}_official_reference_url",
        desc="At least one official reference URL is provided that verifies the stated holiday hours/status",
        parent=candidate_node,
        critical=True,
    )
    if official_sources:
        urls_inline = "; ".join(official_sources)
        ref_claim = (
            f"At least one of these URLs is an official page for '{name}' (brand-owned/corporate site, "
            f"e.g., brand.com or official store-locator) that explicitly mentions {XMAS_EVE_STR} hours, "
            f"{XMAS_DAY_STR} open/closed status, or {NYD_STR} hours: {urls_inline}"
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=official_sources,
            additional_instruction=(
                "A page is 'official' if it is clearly owned/published by the retailer (corporate domain or official store-locator). "
                "Third-party blogs, news, or aggregator sites are NOT official. "
                "The page must mention at least one of the targeted holiday items (Christmas Eve hours, Christmas Day status, or New Year's Day hours)."
            ),
        )
    else:
        # No URLs provided — this must fail
        ref_claim = (
            f"For '{name}', at least one official reference URL verifying the holiday hours/status is provided."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=None,
            additional_instruction=(
                "In the extracted data, the number of provided official URLs is 0 for this chain. "
                "Therefore, the correct judgment is: the claim is NOT supported."
            ),
        )

    return candidate_node


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
    Evaluate an answer for the 'holiday_hours_2025_retail' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Identify at least 4 major U.S. retail store chains that close at 6 p.m. or later on Christmas Eve 2025, "
            "and for each provide: (1) Christmas Eve 2025 closing time, (2) Christmas Day 2025 open/closed status, "
            "(3) New Year's Day 2026 operating hours, and (4) an official reference URL verifying these holiday hours/status."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # 1) Extract structured chains info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=ChainsExtraction,
        extraction_name="chains_extraction",
    )

    # Choose exactly the first 4 candidates (pad if fewer)
    chains4 = _first_k_or_pad(extracted.chains, 4)

    # 2) Build per-candidate verification (parallel groups with critical leaves)
    candidate_nodes: List[VerificationNode] = []
    for i, ch in enumerate(chains4):
        node = await verify_candidate_chain(evaluator, root, ch, i)
        candidate_nodes.append(node)

    # 3) Compute how many candidates fully qualified (all 5 critical leaves passed)
    fully_qualifying = 0
    for cand in candidate_nodes:
        # A candidate is fully qualifying if all its children leaves passed
        all_passed = True
        for child in cand.children:
            # Only consider leaves (they all are leaves here)
            if child.status != "passed":
                all_passed = False
                break
        if all_passed:
            fully_qualifying += 1

    # 4) Add critical gate: At least 4 fully qualifying candidates
    evaluator.add_custom_node(
        result=(fully_qualifying >= 4),
        id="minimum_4_qualifying_chains",
        desc="At least 4 candidate chains are fully qualifying (i.e., for at least 4 candidates, all per-chain critical checks pass).",
        parent=root,
        critical=True,
    )

    # 5) Record custom info for transparency
    evaluator.add_custom_info(
        info={
            "evaluated_candidates": 4,
            "fully_qualifying_count": fully_qualifying,
            "dates": {
                "christmas_eve": XMAS_EVE_STR,
                "christmas_day": XMAS_DAY_STR,
                "new_years_day": NYD_STR,
            },
        },
        info_type="evaluation_meta",
    )

    return evaluator.get_summary()