import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "msg_concert_capacity_2026"
TASK_DESCRIPTION = (
    "What is the seating capacity of Madison Square Garden in New York City for concerts? "
    "Provide the capacity number and include a reference URL from an official or authoritative source."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MSGConcertCapacityExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer.
    """
    venue_name: Optional[str] = None               # e.g., "Madison Square Garden", "MSG"
    location: Optional[str] = None                 # e.g., "New York City", "New York, NY", "NYC"
    configuration: Optional[str] = None            # e.g., "concert", "end-stage concert", "in-the-round"
    capacity_text: Optional[str] = None            # full text mentioning the capacity
    capacity_numeric: Optional[str] = None         # digits-only capacity, e.g., "20000" or "20789"
    as_of_text: Optional[str] = None               # e.g., "as of 2026"
    as_of_year: Optional[str] = None               # e.g., "2026"
    sources: List[str] = Field(default_factory=list)  # list of URLs included in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_msg_capacity() -> str:
    return """
    Extract the following fields from the answer text about Madison Square Garden concert capacity:

    - venue_name: The venue name as written (e.g., "Madison Square Garden", "MSG").
    - location: The stated location for the venue (e.g., "New York City", "New York, NY", "NYC", "Manhattan, New York").
    - configuration: The event configuration that the capacity refers to. If the answer explicitly says it is for concerts, set to "concert" or the closest phrase (e.g., "concert", "end-stage concert", "in-the-round", "center-stage concert"). Otherwise, set to null.
    - capacity_text: The exact text snippet containing the seating capacity number as stated in the answer (e.g., "about 20,000 for concerts").
    - capacity_numeric: If a single specific number is provided for concert capacity, extract that number as digits only (remove commas). If multiple numbers are present, choose the one that is explicitly for concert configuration; otherwise null.
    - as_of_text: The exact phrase indicating recency such as "as of 2026" if present; otherwise null.
    - as_of_year: A 4-digit year mentioned as the currency in the answer (e.g., "2026") if present; otherwise null.
    - sources: All URLs cited in the answer text that are intended to support the capacity. Include only valid URLs explicitly present in the answer.

    Important instructions:
    - Do not invent any values. Only extract what is explicitly present in the answer.
    - For capacity_numeric: If the answer gives a range (e.g., "19,000–20,000") or multiple numbers but does not clearly indicate a single concert capacity number, set capacity_numeric to null.
    - For configuration: Prefer "concert" (or a subtype like "end-stage concert") if the answer states the capacity for concerts; otherwise null.
    - For location: Use the location string as written (normalize nothing).
    - For sources: Extract all URLs exactly as they appear. If the answer lists domains or references without full URLs, ignore those.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_integer_from_text(text: Optional[str]) -> Optional[str]:
    """
    Heuristic to capture a single integer (with optional commas) from a string.
    Returns digits-only string if found, else None.
    """
    if not text:
        return None
    m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d+)\b", text)
    if not m:
        return None
    digits = re.sub(r"[^\d]", "", m.group(1))
    return digits if digits else None


def _looks_like_range(text: Optional[str]) -> bool:
    if not text:
        return False
    # Look for hyphen/en dash or 'to' between numbers
    return bool(re.search(r"\d\s*(?:-|–|—|to)\s*\d", text))


def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if isinstance(u, str) and u.strip() and (u.strip().startswith("http://") or u.strip().startswith("https://")):
            out.append(u.strip())
    return out


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_run_verifications(
    evaluator: Evaluator,
    extracted: MSGConcertCapacityExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and execute checks.
    """
    # Parent critical node that represents the entire rubric item
    main_node = evaluator.add_parallel(
        id="Madison_Square_Garden_Concert_Capacity",
        desc="Verify the answer correctly provides Madison Square Garden (NYC) concert seating capacity with appropriate sourcing and recency per constraints.",
        parent=evaluator.root,
        critical=True,  # All children must be critical as per rule
    )

    # Child 1: Correct venue and location (simple verification from answer)
    node_venue_loc = evaluator.add_leaf(
        id="Correct_Venue_and_Location",
        desc="The answer is explicitly about Madison Square Garden in New York City (not a different venue or location).",
        parent=main_node,
        critical=True,
    )
    claim_venue_loc = (
        "The answer explicitly refers to Madison Square Garden located in New York City (NYC, New York, or equivalent), "
        "and it does not refer to a different venue or location."
    )
    await evaluator.verify(
        claim=claim_venue_loc,
        node=node_venue_loc,
        additional_instruction=(
            "Judge solely from the answer text. Accept common variants such as 'MSG', 'New York City', 'NYC', "
            "'New York, NY', or 'Manhattan' as valid for Madison Square Garden in NYC. "
            "Fail if the answer references a different city or a different venue."
        ),
    )

    # Child 2: Concert configuration specified (simple verification from answer)
    node_concert_cfg = evaluator.add_leaf(
        id="Concert_Configuration_Specified",
        desc="The capacity reported is explicitly for concert configuration (not basketball, hockey, or other event types).",
        parent=main_node,
        critical=True,
    )
    claim_concert_cfg = (
        "The answer explicitly states that the capacity figure is for concerts (concert configuration), "
        "not basketball, hockey, or other events."
    )
    await evaluator.verify(
        claim=claim_concert_cfg,
        node=node_concert_cfg,
        additional_instruction=(
            "Judge solely from the answer text. Accept phrasing like 'for concerts', 'concert capacity', "
            "'end‑stage concert', 'in‑the‑round/center‑stage concert', or similar. "
            "Fail if the answer does not make the concert configuration explicit."
        ),
    )

    # Prepare capacity string to use in claims
    cap_num = extracted.capacity_numeric
    if not cap_num and extracted.capacity_text and not _looks_like_range(extracted.capacity_text):
        cap_num = _first_integer_from_text(extracted.capacity_text)

    # Child 3: Capacity number provided (custom existence check)
    has_single_number = bool(cap_num and cap_num.isdigit())
    # Also make sure the original capacity_text does not clearly indicate a range
    not_range = not _looks_like_range(extracted.capacity_text)
    node_capacity_number = evaluator.add_custom_node(
        result=bool(has_single_number and not_range),
        id="Capacity_Number_Provided",
        desc="The answer provides a specific numeric seating capacity value (a number) for the stated concert configuration.",
        parent=main_node,
        critical=True,
    )

    # Sources list
    srcs = _valid_urls(extracted.sources)

    # Child 4: Authoritative reference URL that supports the stated concert capacity
    node_authoritative = evaluator.add_leaf(
        id="Authoritative_Reference_URL",
        desc="The answer includes at least one verifiable URL from an official or otherwise authoritative source that supports the stated concert capacity.",
        parent=main_node,
        critical=True,
    )
    # If we don't have a number, still attempt verification (it will likely fail), but won't crash.
    cap_for_claim = cap_num if cap_num else (extracted.capacity_text or "the stated value")
    claim_authoritative = (
        f"The concert seating capacity of Madison Square Garden (New York City) is {cap_for_claim}."
    )
    await evaluator.verify(
        claim=claim_authoritative,
        node=node_authoritative,
        sources=srcs,  # If empty, the verifier will safely fail this leaf
        additional_instruction=(
            "Return True only if at least one of the provided URLs is an official or authoritative source AND "
            "explicitly supports the concert capacity value in the claim. "
            "Treat as authoritative: the venue's official sites (e.g., msg.com/thegarden.com), operator/owner pages, "
            "official seating chart or venue spec pages, or widely recognized reputable references (e.g., "
            "major organizations or well‑maintained encyclopedic resources like Wikipedia if the page clearly states "
            "concert capacity for MSG). "
            "If URLs are missing, invalid, not authoritative, or do not explicitly support the concert capacity for concerts, return False."
        ),
    )

    # Child 5: Current as of 2026 (simple verification; answer must indicate currency)
    node_current_2026 = evaluator.add_leaf(
        id="Current_As_of_2026",
        desc="The answer indicates the capacity is the current operational capacity as of 2026 (e.g., explicitly states currency as of 2026 and/or relies on a source that supports that it is current as of 2026).",
        parent=main_node,
        critical=True,
    )
    claim_current_2026 = (
        "The answer indicates that the stated concert capacity is current as of 2026 (e.g., by explicitly saying 'as of 2026' "
        "or equivalent)."
    )
    await evaluator.verify(
        claim=claim_current_2026,
        node=node_current_2026,
        additional_instruction=(
            "Judge from the answer text. Accept explicit statements like 'as of 2026', 'current (2026)', or equivalent. "
            "If the answer does not clearly indicate the capacity is current as of 2026, return False. "
            "Note: Currency must be tied to 2026; earlier years in the answer do not satisfy this requirement."
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
    Entry point to evaluate an agent's answer for the MSG concert capacity task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator; we'll add one critical child node
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

    # 1) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_msg_capacity(),
        template_class=MSGConcertCapacityExtraction,
        extraction_name="extracted_msg_concert_capacity",
    )

    # 2) Build verification tree and run checks
    await build_and_run_verifications(evaluator, extracted)

    # 3) Return summary
    return evaluator.get_summary()