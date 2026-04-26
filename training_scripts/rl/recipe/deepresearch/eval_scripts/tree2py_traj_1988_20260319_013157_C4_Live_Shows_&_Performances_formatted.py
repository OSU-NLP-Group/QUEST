import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rush_2026_highest_capacity_venue"
TASK_DESCRIPTION = """
Rush's 2026 "Fifty Something Tour" includes performances at multiple venues across the United States. Among all the venues scheduled for the US leg of this tour, identify which venue has the highest concert seating capacity. For this venue, provide: (1) The venue name, (2) The concert seating capacity, (3) The year the venue opened, and (4) The complete street address (street number, street name, city, and state).
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """
    Structured extraction for the answer’s selected venue and its supporting references.
    All URLs must be explicitly present in the answer. Leave any list empty if not provided.
    """
    venue_name: Optional[str] = None
    seating_capacity: Optional[str] = None  # Keep as string to allow ranges/formatting
    opening_year: Optional[str] = None
    address: Optional[str] = None  # Expect complete street address including city and state

    # URL sources
    sources_overall: List[str] = Field(default_factory=list)
    sources_tour: List[str] = Field(default_factory=list)            # Tour schedule / announcement pages
    sources_capacity: List[str] = Field(default_factory=list)        # Venue capacity specification pages
    sources_opening_year: List[str] = Field(default_factory=list)    # Historical/opening year sources
    sources_address: List[str] = Field(default_factory=list)         # Official address sources
    sources_highest_claim: List[str] = Field(default_factory=list)   # Any page asserting/consolidating "highest capacity"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_details() -> str:
    return """
    Your task is to extract the single venue that the answer identifies as having the highest concert seating capacity among all US venues on Rush’s 2026 “Fifty Something Tour”, along with the requested details and any supporting URLs explicitly cited in the answer text.

    Extract the following fields:
    - venue_name: The venue selected by the answer as the highest-capacity US tour venue.
    - seating_capacity: The concert seating capacity reported for this venue (keep the exact formatting, such as commas, plus signs, or ranges).
    - opening_year: The year the venue originally opened (4-digit year as written in the answer).
    - address: The complete street address as given in the answer (should include number, street name, city, and state).

    Also extract URLs exactly as they appear in the answer, grouping them as follows (each field is an array of URLs):
    - sources_tour: URLs that show the tour schedule or announcements linking Rush’s 2026 “Fifty Something Tour” to this venue and specifically to a United States date.
    - sources_capacity: URLs that state the venue’s concert seating capacity (prefer official venue site or well-known references).
    - sources_opening_year: URLs that state the venue’s opening year in historical records.
    - sources_address: URLs that state the venue’s official address.
    - sources_highest_claim: URLs that directly assert or allow a clear comparison supporting that this venue has the highest concert seating capacity among all US venues on Rush’s 2026 tour (e.g., a reliable consolidation/comparison page, or any explicit “largest/highest capacity” statement).
    - sources_overall: Any other URLs the answer lists as general references that might support the identification or details.

    Rules:
    - Extract only URLs that are explicitly present in the answer text. Do not invent or infer URLs.
    - If a given category has no URLs, return an empty list for that category.
    - If the answer provides multiple venues, focus only on the single venue that the answer ultimately claims is the highest-capacity US tour venue.
    - Do not normalize values; keep them exactly as stated in the answer (except ensure URLs are complete).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def pick_sources(data: VenueExtraction, ordered_fields: List[str]) -> List[str]:
    """Merge multiple URL fields in order with de-duplication."""
    merged: List[str] = []
    for fname in ordered_fields:
        arr = getattr(data, fname, []) or []
        merged.extend(arr)
    return _dedup_preserve_order(merged)


def with_no_source_policy(base_instruction: str, has_sources: bool) -> str:
    """
    Enforce source-grounding. If no sources are provided, explicitly instruct the judge to mark it Incorrect.
    """
    if has_sources:
        return base_instruction
    return base_instruction + "\n\nIMPORTANT: The answer did not provide any URL for this check; since no source webpage is available, you must return 'Incorrect' (not supported)."


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_highest_capacity_venue(
    evaluator: Evaluator,
    parent_node,
    data: VenueExtraction,
) -> None:
    """
    Build verification leaves per rubric and run the checks.
    """
    # Create main parallel node (as per rubric JSON root)
    main = evaluator.add_parallel(
        id="Rush_2026_Tour_Venue_Analysis",
        desc="Evaluates the identification of the highest-capacity venue on Rush's 2026 US tour and verification of its details",
        parent=parent_node,
        critical=False
    )

    venue = data.venue_name or ""
    capacity = data.seating_capacity or ""
    year = data.opening_year or ""
    address = data.address or ""

    # 1) Venue_Tour_Participation (Critical)
    node_tour = evaluator.add_leaf(
        id="Venue_Tour_Participation",
        desc="Verifies that the identified venue is actually scheduled as part of Rush's 2026 'Fifty Something Tour' in the United States",
        parent=main,
        critical=True
    )
    tour_sources = pick_sources(data, ["sources_tour", "sources_overall"])
    claim_tour = f"{venue} is scheduled as a United States tour stop on Rush's 2026 'Fifty Something Tour'."
    instr_tour = with_no_source_policy(
        "Confirm that the provided page(s) explicitly indicate a 2026 'Fifty Something Tour' date at this exact venue and that the show is in the United States. Reject pages about different years, artists, countries, or venues.",
        has_sources=len(tour_sources) > 0
    )

    # 2) Capacity_Verification (Critical)
    node_capacity = evaluator.add_leaf(
        id="Capacity_Verification",
        desc="Verifies that the reported concert seating capacity matches official venue specifications",
        parent=main,
        critical=True
    )
    capacity_sources = pick_sources(data, ["sources_capacity", "sources_overall"])
    claim_capacity = f"The concert seating capacity of {venue} is {capacity}."
    instr_capacity = with_no_source_policy(
        "Verify the concert configuration capacity (or end‑stage capacity). If multiple capacities appear (e.g., basketball/hockey vs. concert), prefer the concert figure. Allow minor formatting differences (commas, approx, plus signs) but the value should clearly match.",
        has_sources=len(capacity_sources) > 0
    )

    # 3) Highest_Capacity_Claim (Critical)
    node_highest = evaluator.add_leaf(
        id="Highest_Capacity_Claim",
        desc="Verifies that the identified venue has the highest concert seating capacity among all venues on the US leg of Rush's 2026 tour",
        parent=main,
        critical=True
    )
    highest_sources = pick_sources(
        data,
        ["sources_highest_claim", "sources_tour", "sources_capacity", "sources_overall"]
    )
    claim_highest = f"Among all venues on the United States leg of Rush's 2026 'Fifty Something Tour', {venue} has the highest concert seating capacity."
    instr_highest = with_no_source_policy(
        "This is a comparative superlative claim. It must be explicitly supported by a reliable page (e.g., a consolidated comparison or an explicit statement that this venue has the highest concert seating capacity among all US tour venues). Pages that only show one venue's capacity or partial lists are insufficient. If any doubt remains or evidence is insufficient, mark as not supported.",
        has_sources=len(highest_sources) > 0
    )

    # 4) Opening_Year_Accuracy (Critical)
    node_year = evaluator.add_leaf(
        id="Opening_Year_Accuracy",
        desc="Verifies that the provided opening year matches the venue's actual opening date in historical records",
        parent=main,
        critical=True
    )
    year_sources = pick_sources(data, ["sources_opening_year", "sources_overall"])
    claim_year = f"{venue} opened in {year}."
    instr_year = with_no_source_policy(
        "Confirm the original opening year from the page. Do not confuse a renovation or re‑opening with the original opening year unless the page explicitly defines 'opened' accordingly. If the page contradicts the claimed year, mark incorrect.",
        has_sources=len(year_sources) > 0
    )

    # 5) Address_Accuracy (Critical)
    node_addr = evaluator.add_leaf(
        id="Address_Accuracy",
        desc="Verifies that the provided street address matches the venue's official location",
        parent=main,
        critical=True
    )
    addr_sources = pick_sources(data, ["sources_address", "sources_overall"])
    claim_addr = f"The official street address of {venue} is '{address}'."
    instr_addr = with_no_source_policy(
        "Verify that the address exactly corresponds to the official venue location. Minor formatting like 'St.' vs 'Street' is acceptable if the address is the same. The address should include number, street name, city, and state.",
        has_sources=len(addr_sources) > 0
    )

    # 6) Supporting_Documentation (Non-critical existence check)
    all_sources_present = any([
        len(data.sources_overall) > 0,
        len(data.sources_tour) > 0,
        len(data.sources_capacity) > 0,
        len(data.sources_opening_year) > 0,
        len(data.sources_address) > 0,
        len(data.sources_highest_claim) > 0
    ])
    evaluator.add_custom_node(
        result=all_sources_present,
        id="Supporting_Documentation",
        desc="Checks whether reference URLs are provided to support the venue identification and details",
        parent=main,
        critical=False
    )

    # Batch verify (parallel) the factual leaves
    verifications: List[tuple[str, List[str] | None, Any, Optional[str]]] = [
        (claim_tour, tour_sources if tour_sources else None, node_tour, instr_tour),
        (claim_capacity, capacity_sources if capacity_sources else None, node_capacity, instr_capacity),
        (claim_highest, highest_sources if highest_sources else None, node_highest, instr_highest),
        (claim_year, year_sources if year_sources else None, node_year, instr_year),
        (claim_addr, addr_sources if addr_sources else None, node_addr, instr_addr),
    ]

    await evaluator.batch_verify(verifications)


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for Rush’s 2026 highest-capacity US tour venue task.
    Returns a standardized summary dictionary from the evaluator.
    """
    # 1) Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # As per rubric root
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

    # 2) Extract structured info from the answer
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue_details(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # 3) Optional: Record some custom info (e.g., total URLs)
    total_urls = sum([
        len(extracted.sources_overall or []),
        len(extracted.sources_tour or []),
        len(extracted.sources_capacity or []),
        len(extracted.sources_opening_year or []),
        len(extracted.sources_address or []),
        len(extracted.sources_highest_claim or []),
    ])
    evaluator.add_custom_info(
        info={
            "venue_name": extracted.venue_name,
            "seating_capacity": extracted.seating_capacity,
            "opening_year": extracted.opening_year,
            "address": extracted.address,
            "total_urls_cited": total_urls
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    # 4) Build verification tree and run checks
    await verify_highest_capacity_venue(evaluator, root, extracted)

    # 5) Return structured result
    return evaluator.get_summary()