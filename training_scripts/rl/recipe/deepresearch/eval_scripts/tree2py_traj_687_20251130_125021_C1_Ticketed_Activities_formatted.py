import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "every_brilliant_thing_2026_attendance_info"
TASK_DESCRIPTION = (
    'For the upcoming Broadway production of "Every Brilliant Thing" starring Daniel Radcliffe in 2026, '
    "provide the following information for someone planning to attend: (1) the complete street address of the theater "
    "where the show is performed, (2) the show's final performance date, and (3) the show's running time."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AddressInfo(BaseModel):
    full: Optional[str] = None
    street_number: Optional[str] = None
    street_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None


class ProductionExtraction(BaseModel):
    star_name: Optional[str] = None               # e.g., "Daniel Radcliffe"
    production_year: Optional[str] = None         # e.g., "2026"
    broadway_indicator: Optional[str] = None      # e.g., "Broadway", "on Broadway"
    venue_name: Optional[str] = None              # e.g., "Hudson Theatre"
    venue_city: Optional[str] = None              # e.g., "New York" or "New York City"
    venue_state: Optional[str] = None             # e.g., "NY" or "New York"
    address: Optional[AddressInfo] = None
    closing_date: Optional[str] = None            # e.g., "June 28, 2026"
    running_time: Optional[str] = None            # e.g., "75 minutes" or "1 hour 15 minutes"
    urls: List[str] = Field(default_factory=list) # any URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_production_info() -> str:
    return """
    Extract structured information as presented in the answer for the 2026 Broadway production of "Every Brilliant Thing" starring Daniel Radcliffe.

    Extract the following fields exactly as they appear in the answer (do not invent):
    - star_name: The name of the star/lead actor mentioned for this production.
    - production_year: The specific year tied to the production (e.g., 2026) if stated.
    - broadway_indicator: A word or short phrase from the answer that shows this is a Broadway production (e.g., "Broadway", "on Broadway"). If not present, return null.
    - venue_name: The theater name (e.g., "Hudson Theatre").
    - venue_city: City of the venue (e.g., "New York" or "New York City") if stated.
    - venue_state: State of the venue (e.g., "NY" or "New York") if stated.
    - address: The complete street address of the theater, if present. Fill subfields:
        * full: The full address as one string exactly as written in the answer, if present.
        * street_number: e.g., "139"
        * street_name: e.g., "West 44th Street" or "W 44th St"
        * city: e.g., "New York" or "New York City"
        * state: e.g., "NY" or "New York"
        * zip_code: e.g., "10036" or "10036-9999"
      If any subfield is missing from the answer, set it to null. Do not infer.
    - closing_date: The final performance date (closing date) for this production, exactly as written in the answer.
    - running_time: The show's running time duration (e.g., "70 minutes", "1 hour 10 minutes") exactly as written in the answer.
    - urls: List of all URLs explicitly mentioned in the answer text that are relevant to this production (official theater site, production site, Playbill, Broadway, Ticketmaster, press releases, etc.). Only include valid URLs actually present; do not invent.

    If a required piece of information is not explicitly present in the answer, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _compose_address_from_parts(addr: Optional[AddressInfo], fallback_city: Optional[str], fallback_state: Optional[str]) -> Optional[str]:
    if not addr:
        return None

    # Prefer the full address as written in the answer
    if addr.full and addr.full.strip():
        return addr.full.strip()

    # Otherwise compose from parts that exist in the answer
    number = (addr.street_number or "").strip()
    street = (addr.street_name or "").strip()
    city = (addr.city or fallback_city or "").strip()
    state = (addr.state or fallback_state or "").strip()
    zip_code = (addr.zip_code or "").strip()

    left = " ".join([p for p in [number, street] if p])
    right_city_state = ", ".join([p for p in [city, state] if p])

    pieces = []
    if left:
        pieces.append(left)
    if right_city_state:
        pieces.append(right_city_state)
    if zip_code:
        pieces[-1] = pieces[-1] + f" {zip_code}" if pieces else zip_code

    composed = ", ".join(pieces) if pieces else None
    return composed.strip() if composed else None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root_node, extracted: ProductionExtraction) -> None:
    """
    Build verification nodes according to the rubric and run verifications.
    The rubric requires five critical checks under a critical, parallel parent.
    """

    # Create a critical, parallel node that mirrors the rubric's top-level requirement
    info_node = evaluator.add_parallel(
        id="Every_Brilliant_Thing_Information",
        desc="Provides required attendance information for the specified Broadway production of “Every Brilliant Thing” starring Daniel Radcliffe (2026).",
        parent=root_node,
        critical=True
    )

    # 1) Production_Context_Match (Critical leaf)
    n_context = evaluator.add_leaf(
        id="Production_Context_Match",
        desc="Answer indicates the information pertains to the 2026 Broadway production of “Every Brilliant Thing” starring Daniel Radcliffe.",
        parent=info_node,
        critical=True
    )
    context_claim = (
        "The answer clearly indicates that the provided information pertains to the 2026 Broadway production "
        "of 'Every Brilliant Thing' starring Daniel Radcliffe (i.e., it mentions Daniel Radcliffe, the year 2026, "
        "and that it is a Broadway production)."
    )
    await evaluator.verify(
        claim=context_claim,
        node=n_context,
        additional_instruction=(
            "Judge this only based on the answer text. Accept minor variations such as 'on Broadway', "
            "'Broadway run', or equivalent phrasing, and allow different placements of the year."
        ),
    )

    # 2) Venue_Requirement (Critical leaf)
    n_venue = evaluator.add_leaf(
        id="Venue_Requirement",
        desc="Identifies the venue as Hudson Theatre in New York City.",
        parent=info_node,
        critical=True
    )
    venue_claim = (
        "The answer identifies the venue for this production as the Hudson Theatre in New York City (NYC)."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=n_venue,
        additional_instruction=(
            "Judge this only based on the answer text. Accept minor variants like 'Hudson Theater' vs 'Hudson Theatre', "
            "and 'New York' vs 'New York City'."
        ),
    )

    # Prepare address string and sources (if any)
    address_str = _compose_address_from_parts(
        extracted.address,
        fallback_city=extracted.venue_city,
        fallback_state=extracted.venue_state
    )
    sources_list = extracted.urls or []

    # 3) Theater_Address (Critical leaf)
    n_address = evaluator.add_leaf(
        id="Theater_Address",
        desc="Provides Hudson Theatre’s complete street address, including street number, street name, city, state, and ZIP code.",
        parent=info_node,
        critical=True
    )
    addr_claim = (
        f"The answer provides Hudson Theatre’s complete street address as: '{address_str}'. "
        "This address includes the street number, street name, city, state, and ZIP code."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=n_address,
        sources=sources_list,
        additional_instruction=(
            "Verify either from the answer text or from the provided URLs that the address is present and complete. "
            "Allow common formatting variants (e.g., 'W' vs 'West', abbreviations, commas). "
            "ZIP may be 5 or 9 digits (ZIP+4). If verifying via URL, ensure the page clearly shows the same complete address."
        ),
    )

    # 4) Closing_Date (Critical leaf)
    n_closing = evaluator.add_leaf(
        id="Closing_Date",
        desc="Provides the show’s final performance date (closing date) for this production.",
        parent=info_node,
        critical=True
    )
    closing_claim = (
        f"The final performance (closing) date for this 2026 Broadway production is '{extracted.closing_date}'."
    )
    await evaluator.verify(
        claim=closing_claim,
        node=n_closing,
        sources=sources_list,
        additional_instruction=(
            "If the answer states a run (e.g., 'through June 28, 2026'), treat the last day as the closing date. "
            "When URLs are provided, confirm the closing date against the page; otherwise, judge based on the answer text."
        ),
    )

    # 5) Running_Time (Critical leaf)
    n_runtime = evaluator.add_leaf(
        id="Running_Time",
        desc="Provides the show’s running time duration for this production.",
        parent=info_node,
        critical=True
    )
    runtime_claim = f"The running time (duration) of the show for this production is '{extracted.running_time}'."
    await evaluator.verify(
        claim=runtime_claim,
        node=n_runtime,
        sources=sources_list,
        additional_instruction=(
            "Accept reasonable duration equivalents (e.g., '70 minutes' ~ '1h10m' ~ '1 hour 10 minutes', "
            "'approx.' qualifiers). Prefer verifying with URLs when available; otherwise judge based on the answer."
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
    Evaluate an answer for the 2026 Broadway 'Every Brilliant Thing' attendance info task.
    """
    # Initialize evaluator with a parallel root
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

    # Extract production info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_production_info(),
        template_class=ProductionExtraction,
        extraction_name="production_info",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()