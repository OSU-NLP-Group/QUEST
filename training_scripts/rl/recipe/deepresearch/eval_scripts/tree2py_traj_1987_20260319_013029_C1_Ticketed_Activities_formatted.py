import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "msg_vs_rcmh_concert_capacity"
TASK_DESCRIPTION = """
Between Madison Square Garden and Radio City Music Hall in New York City, which venue has the larger seating capacity for concerts?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ComparisonExtraction(BaseModel):
    """
    Extract what the answer compared and concluded, plus any cited sources.
    """
    mentioned_venues: List[str] = Field(default_factory=list, description="Venue names the answer explicitly compares")
    capacity_context: Optional[str] = Field(default=None, description="What capacity context the answer used (e.g., 'concert seating capacity')")
    larger_venue: Optional[str] = Field(default=None, description="Which venue the answer claims is larger for concerts, if stated")
    msg_concert_capacity: Optional[str] = Field(default=None, description="MSG concert seating capacity as stated in the answer (string as-is)")
    rcmh_concert_capacity: Optional[str] = Field(default=None, description="Radio City Music Hall concert seating capacity as stated in the answer (string as-is)")
    msg_capacity_urls: List[str] = Field(default_factory=list, description="URLs cited for MSG concert capacity")
    rcmh_capacity_urls: List[str] = Field(default_factory=list, description="URLs cited for Radio City Music Hall concert capacity")
    msg_events_2026_urls: List[str] = Field(default_factory=list, description="URLs cited that show MSG has ticketed concert events scheduled for 2026")
    rcmh_events_2026_urls: List[str] = Field(default_factory=list, description="URLs cited that show Radio City Music Hall has ticketed concert events scheduled for 2026")
    other_urls: List[str] = Field(default_factory=list, description="Any other URLs cited that are relevant to the venues")


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_comparison() -> str:
    return """
    Extract from the answer the comparison details between Madison Square Garden (also referred to as "MSG") and Radio City Music Hall (also referred to as "Radio City").
    Return a JSON object with the following fields:
    - mentioned_venues: list of venue names that the answer explicitly compares (e.g., ["Madison Square Garden", "Radio City Music Hall"])
    - capacity_context: the type of capacity discussed (e.g., "concert seating capacity", "basketball capacity", "maximum capacity"). If it's about concerts, return a phrase that clearly contains "concert".
    - larger_venue: which venue the answer claims is larger for the specified capacity context (return the venue name exactly as stated; if not stated, null)
    - msg_concert_capacity: the concert seating capacity number for Madison Square Garden as stated in the answer (string as-is; if not present, null)
    - rcmh_concert_capacity: the concert seating capacity number for Radio City Music Hall as stated in the answer (string as-is; if not present, null)
    - msg_capacity_urls: list of URLs cited that support MSG concert seating capacity (if none, return an empty list)
    - rcmh_capacity_urls: list of URLs cited that support Radio City Music Hall concert seating capacity (if none, return an empty list)
    - msg_events_2026_urls: list of URLs cited that show MSG has ticketed concert events scheduled for 2026 (if none, return an empty list)
    - rcmh_events_2026_urls: list of URLs cited that show Radio City Music Hall has ticketed concert events scheduled for 2026 (if none, return an empty list)
    - other_urls: any other URLs cited in the answer that are relevant to the two venues (if none, return an empty list)

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer (including plain URLs or markdown links). Do not invent URLs.
    - Do not normalize venue names; extract exactly as written.
    - Prefer strings for capacities, not numbers.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls or []:
        if not u:
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


def _union_urls(*url_lists: List[str]) -> List[str]:
    all_urls: List[str] = []
    for lst in url_lists:
        all_urls.extend(lst or [])
    return _dedup(all_urls)


# --------------------------------------------------------------------------- #
# Verification building                                                       #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, ext: ComparisonExtraction) -> None:
    """
    Build and execute the verification tree according to the rubric.
    We introduce a critical parent node to mirror the rubric's critical root.
    """
    # Critical top-level node (parallel aggregation)
    task_node = evaluator.add_parallel(
        id="determine_larger_concert_capacity",
        desc="Evaluate whether the answer correctly compares the two specified venues and identifies which has the larger seating capacity for concerts, while respecting the stated constraints.",
        parent=root,
        critical=True
    )

    # 1) Uses the specified venues (leaf)
    uses_specified_leaf = evaluator.add_leaf(
        id="uses_specified_venues",
        desc="Answer compares Madison Square Garden and Radio City Music Hall (not different venues).",
        parent=task_node,
        critical=True
    )
    claim_uses_specified = (
        "The answer compares Madison Square Garden (also acceptable: 'MSG') and Radio City Music Hall "
        "(also acceptable: 'Radio City'). Mentions of other venues for context are acceptable as long as the "
        "core comparison is between these two."
    )

    # 2) Uses concert seating capacity context (leaf)
    uses_concert_context_leaf = evaluator.add_leaf(
        id="uses_concert_seating_context",
        desc="Answer addresses seating capacity specifically for concerts (not unrelated capacities/configurations).",
        parent=task_node,
        critical=True
    )
    claim_uses_concert_context = (
        "The answer explicitly addresses concert seating capacity (or concert configuration capacity), "
        "not basketball, hockey, generic maximum capacity, or floor-standing GA counts."
    )

    # 3) Satisfies venue eligibility constraints (critical parallel with two leaves)
    eligibility_node = evaluator.add_parallel(
        id="eligibility_constraints",
        desc="The venues being compared satisfy the stated eligibility constraints (major concert venues in NYC; have ticketed concert events scheduled for 2026).",
        parent=task_node,
        critical=True
    )

    # 3.a) Major concert venues in NYC
    nyc_major_leaf = evaluator.add_leaf(
        id="nyc_major_concert_venues",
        desc="Madison Square Garden and Radio City Music Hall are major concert venues located in New York City.",
        parent=eligibility_node,
        critical=True
    )
    claim_nyc_major = (
        "Madison Square Garden and Radio City Music Hall are major concert venues and are located in New York City."
    )
    nyc_major_sources = _union_urls(ext.msg_capacity_urls, ext.rcmh_capacity_urls, ext.other_urls)

    # 3.b) Have ticketed concert events scheduled for 2026
    events_2026_leaf = evaluator.add_leaf(
        id="events_scheduled_2026",
        desc="Madison Square Garden and Radio City Music Hall each have ticketed concert events scheduled for 2026.",
        parent=eligibility_node,
        critical=True
    )
    claim_events_2026 = (
        "Both Madison Square Garden and Radio City Music Hall each have at least one ticketed concert event scheduled for 2026."
    )
    events_2026_sources = _union_urls(ext.msg_events_2026_urls, ext.rcmh_events_2026_urls)

    # 4) Identifies the correct larger-capacity venue
    # We split this into two critical sequential checks:
    #   a) The answer states MSG is larger for concert seating capacity.
    #   b) This fact is supported by cited sources (capacity sources).
    larger_seq = evaluator.add_sequential(
        id="larger_capacity_verification",
        desc="Answer identifies Madison Square Garden as having the larger concert seating capacity, and this is supported by sources.",
        parent=task_node,
        critical=True
    )

    answer_claims_msg_larger_leaf = evaluator.add_leaf(
        id="answer_claims_msg_larger",
        desc="The answer states that Madison Square Garden has a larger concert seating capacity than Radio City Music Hall.",
        parent=larger_seq,
        critical=True
    )
    claim_answer_says_msg_larger = (
        "The answer states that Madison Square Garden (MSG) has a larger concert seating capacity than Radio City Music Hall."
    )

    supported_by_sources_leaf = evaluator.add_leaf(
        id="msg_larger_supported_by_sources",
        desc="It is factually correct that Madison Square Garden has a larger concert seating capacity than Radio City Music Hall, supported by the cited sources.",
        parent=larger_seq,
        critical=True
    )
    fact_claim_msg_larger = (
        "Madison Square Garden has a larger concert seating capacity than Radio City Music Hall."
    )
    capacity_sources = _union_urls(ext.msg_capacity_urls, ext.rcmh_capacity_urls)

    # Prepare batch verifications for parallelizable leaves
    batch_items: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # uses specified venues (simple; no external sources)
    batch_items.append((
        claim_uses_specified,
        None,
        uses_specified_leaf,
        "Consider 'MSG' equivalent to 'Madison Square Garden' and 'Radio City' equivalent to 'Radio City Music Hall'. "
        "It is fine if other venues are mentioned; the key is that these two are the ones being directly compared."
    ))

    # uses concert seating capacity context (simple)
    batch_items.append((
        claim_uses_concert_context,
        None,
        uses_concert_context_leaf,
        "Look for phrases like 'concert seating capacity', 'concert configuration', or equivalent. "
        "Reject comparisons that are clearly about sports seating or generic max occupancy not tied to concerts."
    ))

    # major NYC venues (prefer URL verification if available)
    batch_items.append((
        claim_nyc_major,
        nyc_major_sources if nyc_major_sources else None,
        nyc_major_leaf,
        "Confirm that both venues are located in New York City and commonly operate as major concert venues."
    ))

    # 2026 events scheduled (prefer URL verification if available)
    batch_items.append((
        claim_events_2026,
        events_2026_sources if events_2026_sources else None,
        events_2026_leaf,
        "Use schedule, events, or ticketing pages to confirm at least one 2026 concert at each venue. "
        "If dates are displayed in a calendar or event list, that is sufficient."
    ))

    # answer explicitly claims MSG is larger (simple check against answer text)
    batch_items.append((
        claim_answer_says_msg_larger,
        None,
        answer_claims_msg_larger_leaf,
        "Treat 'MSG' as 'Madison Square Garden'. Minor wording variations are acceptable as long as the claim is clear."
    ))

    # Execute all batch-verifiable checks
    await evaluator.batch_verify(batch_items)

    # Finally, verify the factual larger-capacity claim with sources if available
    await evaluator.verify(
        claim=fact_claim_msg_larger,
        node=supported_by_sources_leaf,
        sources=capacity_sources if capacity_sources else None,
        additional_instruction=(
            "Focus on concert seating capacity (concert configuration). Accept small numeric variations or ranges. "
            "Common reference values are approximately MSG ~19,000–19,500 for concerts and Radio City Music Hall ~5,900–6,000 for concerts."
        )
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
    Evaluate an answer for the MSG vs Radio City Music Hall concert seating capacity comparison.
    """
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
        default_model=model,
    )

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_comparison(),
        template_class=ComparisonExtraction,
        extraction_name="comparison_extraction",
    )

    # Ground truth information (for transparency only; not used to directly judge)
    evaluator.add_ground_truth({
        "expected_larger_venue": "Madison Square Garden",
        "typical_concert_capacities": {
            "Madison Square Garden": "around 19,000–19,500 for concerts",
            "Radio City Music Hall": "around 5,900–6,000 for concerts"
        },
        "eligibility_constraints": [
            "Both venues are major concert venues in New York City",
            "Each has at least one ticketed concert event scheduled for 2026"
        ]
    })

    # Build and run verification tree
    await build_verification_tree(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()