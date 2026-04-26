import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# =============================================================================
# Task constants
# =============================================================================
TASK_ID = "nds_2025_best_in_show"
TASK_DESCRIPTION = (
    "For the 2025 National Dog Show that aired on Thanksgiving Day, provide the following "
    "information about the Best in Show winner: (1) the dog's name and breed, "
    "(2) the handler's full name and the city and state where the handler is based, "
    "(3) the television network that broadcast the show, "
    "(4) the specific venue name and its location (city and state) where the event took place, "
    "(5) which of the seven groups the dog won before being named Best in Show, "
    "and (6) the monetary prize amount awarded to the winner."
)

# Grounded expectations as per rubric (recorded for transparency/debug)
EXPECTED_FACTS = {
    "winner_name": "Soleil",
    "breed": "Belgian Sheepdog",
    "handler_name": "Daniel Martin",
    "handler_city": "Princeton",
    "handler_state": "North Carolina",
    "event_venue_name": "Greater Philadelphia Expo Center",
    "event_venue_city": "Oaks",
    "event_venue_state": "Pennsylvania",
    "event_host": "Kennel Club of Philadelphia",
    "broadcast_network": "NBC",
    "broadcast_date": "November 27, 2025",
    "broadcast_time": "12 p.m. to 2 p.m. ET",
    "group_before_bis": "Herding Group",
    "prize_amount": "$2,000",
    "prize_items": ["Purina Pro Plan embroidered chair", "Yeti dog bowl"],
}


# =============================================================================
# Extraction models
# =============================================================================
class WinnerExtraction(BaseModel):
    dog_name: Optional[str] = None
    breed: Optional[str] = None
    age: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HandlerExtraction(BaseModel):
    full_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BroadcastExtraction(BaseModel):
    network: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HostExtraction(BaseModel):
    host_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GroupExtraction(BaseModel):
    group_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PrizeExtraction(BaseModel):
    amount: Optional[str] = None
    items: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class NDS2025Extraction(BaseModel):
    winner: Optional[WinnerExtraction] = None
    handler: Optional[HandlerExtraction] = None
    broadcast: Optional[BroadcastExtraction] = None
    venue: Optional[VenueExtraction] = None
    host: Optional[HostExtraction] = None
    group: Optional[GroupExtraction] = None
    prize: Optional[PrizeExtraction] = None
    general_sources: List[str] = Field(default_factory=list)


# =============================================================================
# Extraction prompt
# =============================================================================
def prompt_extract_all() -> str:
    return """
Extract structured information about the 2025 National Dog Show (the Thanksgiving Day televised broadcast) from the provided answer text.

Return a single JSON object matching this schema:
- winner:
  - dog_name: the dog's name
  - breed: the dog's breed, exactly as stated (e.g., "Belgian Sheepdog" or "Belgian Shepherd (Groenendael)")
  - age: the dog's age as written (e.g., "6 years old" or "6")
  - sources: array of all URLs cited in the answer that directly support the winner identity/age/breed
- handler:
  - full_name: handler's full name
  - city: city where the handler is based
  - state: state where the handler is based
  - sources: array of URLs that support the handler's identity/location
- broadcast:
  - network: the TV network that broadcast the show
  - date: the broadcast date as given (e.g., "November 27, 2025")
  - time: the broadcast time window as given (e.g., "12 p.m. to 2 p.m. ET")
  - sources: array of URLs that support the broadcast details
- venue:
  - venue_name: the specific venue name
  - city: the venue's city
  - state: the venue's state
  - sources: array of URLs that support the venue details
- host:
  - host_name: the event host/organizer name (e.g., Kennel Club of Philadelphia)
  - sources: array of URLs that support the host identity
- group:
  - group_name: which of the seven groups the dog won before Best in Show (e.g., "Herding Group")
  - sources: array of URLs that support the group result
- prize:
  - amount: the monetary prize amount (as written, include the currency symbol if present)
  - items: array of any non-cash items mentioned (e.g., "Purina Pro Plan embroidered chair", "Yeti dog bowl")
  - sources: array of URLs that support the prize information
- general_sources:
  - array of any other URLs in the answer that are relevant to the overall event/winner if not already captured above

Extraction rules:
1) Extract only what is explicitly stated in the answer. If a field is missing, set it to null (or [] for arrays).
2) For URLs, extract only valid, explicit URLs (including those present in markdown links). If none, return [].
3) Do not hallucinate or infer facts; preserve wording as in the answer when possible.
"""


# =============================================================================
# Helper utilities
# =============================================================================
def _combine_sources(*source_lists: Optional[List[str]]) -> Optional[List[str]]:
    merged: List[str] = []
    seen = set()
    for sl in source_lists:
        if not sl:
            continue
        for u in sl:
            if isinstance(u, str) and u.strip() and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged if merged else None


async def _add_and_verify(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]],
    add_ins: str,
    critical: bool = True,
):
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins,
    )


# =============================================================================
# Verification orchestration
# =============================================================================
async def _build_and_verify_checks(evaluator: Evaluator, extracted: NDS2025Extraction) -> None:
    # Create a critical parallel node to represent the rubric root (all items mandatory)
    rubric_root = evaluator.add_parallel(
        id="nds_2025_checks",
        desc="Verify all required information about the 2025 National Dog Show (Thanksgiving Day broadcast) Best in Show winner per the rubric",
        parent=evaluator.root,
        critical=True,
    )

    # Prepare per-field sources with fallback to general_sources
    general_sources = (extracted.general_sources or []) if extracted else []

    winner_sources = _combine_sources(
        extracted.winner.sources if extracted and extracted.winner else [],
        general_sources,
    )

    handler_sources = _combine_sources(
        extracted.handler.sources if extracted and extracted.handler else [],
        general_sources,
    )

    venue_sources = _combine_sources(
        extracted.venue.sources if extracted and extracted.venue else [],
        general_sources,
    )

    host_sources = _combine_sources(
        extracted.host.sources if extracted and extracted.host else [],
        general_sources,
    )

    broadcast_sources = _combine_sources(
        extracted.broadcast.sources if extracted and extracted.broadcast else [],
        general_sources,
    )

    group_sources = _combine_sources(
        extracted.group.sources if extracted and extracted.group else [],
        general_sources,
    )

    prize_sources = _combine_sources(
        extracted.prize.sources if extracted and extracted.prize else [],
        general_sources,
    )

    # 1) Winning dog identity
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="winning_dog_identity",
        desc="Best in Show winner is identified as Soleil, a Belgian Sheepdog",
        claim="At the 2025 National Dog Show (the Thanksgiving Day broadcast), the Best in Show winner was named Soleil and is a Belgian Sheepdog (aka Belgian Shepherd, Groenendael variety).",
        sources=winner_sources,
        add_ins="Accept breed synonyms such as 'Belgian Shepherd' or 'Groenendael' as equivalent to 'Belgian Sheepdog'. Focus on the 2025 Thanksgiving broadcast event.",
        critical=True,
    )

    # 2) Dog age
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="dog_age",
        desc="Soleil's age is identified as 6 years old",
        claim="The Best in Show winner, Soleil, was 6 years old at the time of the 2025 National Dog Show Thanksgiving broadcast.",
        sources=winner_sources,
        add_ins="Minor phrasing variations like '6' or 'six years old' should be treated as equivalent.",
        critical=True,
    )

    # 3) Handler name
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="handler_name",
        desc="Handler is identified as Daniel Martin",
        claim="Soleil's handler was Daniel Martin.",
        sources=handler_sources,
        add_ins="Allow minor formatting variations (e.g., middle initials).",
        critical=True,
    )

    # 4) Handler location
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="handler_location",
        desc="Handler is based in Princeton, North Carolina",
        claim="Soleil's handler, Daniel Martin, is based in Princeton, North Carolina.",
        sources=handler_sources,
        add_ins="Treat 'Princeton, NC' as equivalent to 'Princeton, North Carolina'.",
        critical=True,
    )

    # 5) Event venue
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="event_venue",
        desc="Event venue is identified as the Greater Philadelphia Expo Center in Oaks, Pennsylvania",
        claim="The event took place at the Greater Philadelphia Expo Center in Oaks, Pennsylvania.",
        sources=venue_sources,
        add_ins="Accept 'Oaks, PA' as equivalent to 'Oaks, Pennsylvania'.",
        critical=True,
    )

    # 6) Event host
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="event_host",
        desc="Event host is identified as the Kennel Club of Philadelphia",
        claim="The event is hosted by the Kennel Club of Philadelphia.",
        sources=host_sources or venue_sources,  # host often appears with venue information
        add_ins="Acronyms like 'KCP' refer to Kennel Club of Philadelphia.",
        critical=True,
    )

    # 7) Broadcast network
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="broadcast_network",
        desc="Broadcast network is identified as NBC",
        claim="The 2025 National Dog Show Thanksgiving broadcast aired on NBC.",
        sources=broadcast_sources,
        add_ins="The show is traditionally broadcast by NBC on Thanksgiving Day; verify this year's network explicitly.",
        critical=True,
    )

    # 8) Broadcast date
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="broadcast_date",
        desc="Broadcast date is identified as November 27, 2025 (Thanksgiving Day)",
        claim="The broadcast date was November 27, 2025 (Thanksgiving Day).",
        sources=broadcast_sources,
        add_ins="Verify the specific date corresponds to Thanksgiving Day in 2025.",
        critical=True,
    )

    # 9) Broadcast time
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="broadcast_time",
        desc="Broadcast time is identified as 12 p.m. to 2 p.m. ET",
        claim="The broadcast time window was from 12 p.m. to 2 p.m. Eastern Time.",
        sources=broadcast_sources,
        add_ins="Equivalent phrasings like 'noon–2 pm ET' or '12-2 pm ET' should count as the same time window.",
        critical=True,
    )

    # 10) Group won before BIS
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="group_won",
        desc="Group won before Best in Show is identified as the Herding Group",
        claim="Before being named Best in Show, Soleil won the Herding Group.",
        sources=group_sources or winner_sources,
        add_ins="Consider synonymous phrasings like 'took the Herding Group' or 'won Herding'.",
        critical=True,
    )

    # 11) Prize package
    await _add_and_verify(
        evaluator=evaluator,
        parent=rubric_root,
        node_id="prize_package",
        desc="Prize package includes $2,000, a Purina Pro Plan embroidered chair, and a Yeti dog bowl",
        claim="The Best in Show prize package included $2,000, a Purina Pro Plan embroidered chair, and a Yeti dog bowl.",
        sources=prize_sources,
        add_ins="All three components ($2,000, embroidered chair, and Yeti dog bowl) must be present to count as supported.",
        critical=True,
    )


# =============================================================================
# Main evaluation entry point
# =============================================================================
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
    Evaluate an answer for the 2025 National Dog Show (Thanksgiving broadcast) Best in Show rubric.
    """
    # Initialize evaluator
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

    # Record expected facts for transparency (not used directly for scoring)
    evaluator.add_ground_truth(EXPECTED_FACTS, gt_type="expected_facts")

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=NDS2025Extraction,
        extraction_name="extracted_answer_facts",
    )

    # Build verification nodes and run checks
    await _build_and_verify_checks(evaluator, extracted)

    # Return summarized evaluation result
    return evaluator.get_summary()