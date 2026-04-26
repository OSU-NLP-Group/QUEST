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
TASK_ID = "us_music_festivals_june_2025"
TASK_DESCRIPTION = (
    "Identify two multi-day music festivals in the United States that take place during June 2025. For each festival, provide the following information:\n\n"
    "1. Festival Name and Official Website: The official name of the festival and a link to its official website.\n\n"
    "2. Duration and Dates: Confirmation that the festival runs for at least 3 consecutive days, along with the specific dates (including day, month, and year) the festival takes place.\n\n"
    "3. Venue Address: The complete physical address of the festival venue, including street address, city, state, and ZIP code.\n\n"
    "4. Camping Options: Confirmation that the festival offers on-site camping options (such as tent camping or RV camping), along with a link to the official camping information page.\n\n"
    "5. Ticket Information: A link to the official ticket purchasing page and confirmation that the festival offers payment plan or layaway options for ticket purchases."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FestivalItem(BaseModel):
    """Single festival information extracted from the answer."""
    name: Optional[str] = None
    website_url: Optional[str] = None

    # Dates
    dates_text: Optional[str] = None  # e.g., "June 6–9, 2025" or "June 20-23, 2025"
    dates_url: Optional[str] = None   # page where dates/schedule is shown (if provided)

    # Venue
    venue_address: Optional[str] = None  # full US postal address
    venue_url: Optional[str] = None      # page where address is shown (if provided)

    # Camping
    camping_info_url: Optional[str] = None

    # Tickets
    tickets_url: Optional[str] = None


class FestivalsExtraction(BaseModel):
    """Top-level extraction of festivals."""
    festivals: List[FestivalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals() -> str:
    return """
    Extract up to TWO music festivals described in the answer that take place in the United States during June 2025.
    For each festival, return an object with the following fields, exactly as provided in the answer (do not invent):
    - name: The official name of the festival (string).
    - website_url: The official festival website URL (string). If missing, return null.
    - dates_text: The specific event dates text as stated in the answer (e.g., "June 6–9, 2025"). If missing, return null.
    - dates_url: A URL that shows the dates/schedule if explicitly provided in the answer; otherwise null.
    - venue_address: The complete physical address of the venue (street, city, state, ZIP) as provided in the answer. If incomplete or missing, return null.
    - venue_url: A URL that shows the venue/address information if explicitly provided in the answer; otherwise null.
    - camping_info_url: A URL to the official camping information page if explicitly provided in the answer; otherwise null.
    - tickets_url: A URL to the official ticket purchasing page if explicitly provided in the answer; otherwise null.

    Rules:
    - If the answer lists more than two festivals, extract only the first two.
    - Extract only URLs that are explicitly present in the answer (including markdown links). Do not infer or fabricate links.
    - If a URL is missing a protocol, prepend "http://".
    - If some fields are not mentioned, set them to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*urls: Optional[str]) -> Optional[List[str] | str]:
    """Return a single URL if exactly one is non-empty; a list if multiple; or None if none."""
    valid = [u for u in urls if isinstance(u, str) and u.strip()]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return valid


def _is_filled(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


# --------------------------------------------------------------------------- #
# Verification logic per festival                                             #
# --------------------------------------------------------------------------- #
async def verify_single_festival(
    evaluator: Evaluator,
    parent_node,
    festival: FestivalItem,
    index: int
) -> None:
    """
    Build verification subtree for a single festival.
    """
    fest_num = index + 1
    fest_node = evaluator.add_parallel(
        id=f"festival_{index}",
        desc=f"Complete information about the {'first' if index == 0 else 'second'} qualifying music festival",
        parent=parent_node,
        critical=False,
    )

    # 1) Festival Name & Official Website (critical): Provided
    name_site_ok = _is_filled(festival.name) and _is_filled(festival.website_url)
    evaluator.add_custom_node(
        result=name_site_ok,
        id=f"festival_{index}_name_website",
        desc=f"The {'first' if index == 0 else 'second'} festival's official name and official festival website URL are provided",
        parent=fest_node,
        critical=True
    )

    # 2) Duration & Dates (critical): At least 3 consecutive days in June 2025, specific dates provided
    dates_leaf = evaluator.add_leaf(
        id=f"festival_{index}_duration_dates",
        desc=f"The {'first' if index == 0 else 'second'} festival is a multi-day event (at least 3 consecutive days) with specific dates provided in June 2025",
        parent=fest_node,
        critical=True
    )

    dates_sources = _combine_sources(festival.dates_url, festival.website_url)
    date_text_for_claim = festival.dates_text if _is_filled(festival.dates_text) else "N/A"
    dates_claim = (
        f"The festival '{festival.name or 'N/A'}' takes place over at least 3 consecutive days in June 2025, "
        f"with specific dates: {date_text_for_claim}."
    )
    dates_instruction = (
        "Only pass if the provided webpage(s) explicitly show the event dates occurring in June 2025 and spanning "
        "at least three consecutive calendar days (e.g., June 6–9, 2025). If the answer does not state specific dates, "
        "or the dates are not clearly in June 2025, or cover fewer than 3 days, mark as NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_leaf,
        sources=dates_sources,
        additional_instruction=dates_instruction
    )

    # 3) Venue Address (critical): complete physical address
    venue_leaf = evaluator.add_leaf(
        id=f"festival_{index}_venue_address",
        desc=f"The {'first' if index == 0 else 'second'} festival's venue has a complete physical address (street address, city, state, ZIP code) provided",
        parent=fest_node,
        critical=True
    )

    venue_sources = _combine_sources(festival.venue_url, festival.website_url)
    venue_addr_for_claim = festival.venue_address if _is_filled(festival.venue_address) else "N/A"
    venue_claim = (
        f"The festival venue address is: {venue_addr_for_claim}. "
        f"This address includes street address, city, state, and a 5-digit ZIP code in the United States."
    )
    venue_instruction = (
        "Only pass if the evidence page clearly shows a full US postal address that includes street address, city, state, "
        "and a ZIP code. If the address is missing any of these components or the answer did not provide a complete address, "
        "mark as NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=venue_sources,
        additional_instruction=venue_instruction
    )

    # 4) Camping Options (critical): on-site camping + a link to camping info page
    camping_leaf = evaluator.add_leaf(
        id=f"festival_{index}_camping",
        desc=f"The {'first' if index == 0 else 'second'} festival offers on-site camping options (tent camping and/or RV camping) with a link to camping information",
        parent=fest_node,
        critical=True
    )

    camping_sources = _combine_sources(festival.camping_info_url, festival.website_url)
    camping_url_for_claim = festival.camping_info_url if _is_filled(festival.camping_info_url) else "N/A"
    camping_claim = (
        f"The festival offers on-site camping (tent and/or RV). The official camping information page is {camping_url_for_claim}."
    )
    camping_instruction = (
        "Only pass if the evidence page explicitly indicates that on-site camping is available (tent and/or RV). "
        "Additionally, the answer must provide a dedicated official camping information URL; if such a URL is missing or null, "
        "mark as NOT SUPPORTED even if camping is mentioned elsewhere."
    )
    await evaluator.verify(
        claim=camping_claim,
        node=camping_leaf,
        sources=camping_sources,
        additional_instruction=camping_instruction
    )

    # 5) Ticket Information (critical): official ticket purchasing page + payment plan/layaway options
    tickets_leaf = evaluator.add_leaf(
        id=f"festival_{index}_tickets",
        desc=f"The {'first' if index == 0 else 'second'} festival has an official ticket purchasing page link and offers payment plan or layaway options",
        parent=fest_node,
        critical=True
    )

    tickets_sources = _combine_sources(festival.tickets_url, festival.website_url)
    tickets_url_for_claim = festival.tickets_url if _is_filled(festival.tickets_url) else "N/A"
    tickets_claim = (
        f"The festival has an official ticket purchasing page at {tickets_url_for_claim} and offers a payment plan or layaway option for ticket purchases."
    )
    tickets_instruction = (
        "Only pass if the evidence page is an official ticket purchasing page (or directly linked official ticket portal) "
        "and explicitly mentions payment plan or layaway options (e.g., installments, deposit plans, Affirm, Klarna). "
        "If the answer did not provide a ticket purchasing URL, mark as NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=tickets_claim,
        node=tickets_leaf,
        sources=tickets_sources,
        additional_instruction=tickets_instruction
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
    Evaluate an answer for the 'two multi-day US music festivals in June 2025' task.
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
        default_model=model
    )

    # Extract up to 2 festivals from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_festivals(),
        template_class=FestivalsExtraction,
        extraction_name="festivals_extraction"
    )

    # Ensure exactly two slots (pad with empty items if fewer)
    festivals: List[FestivalItem] = list(extracted.festivals[:2])
    while len(festivals) < 2:
        festivals.append(FestivalItem())

    # Create two parallel festival verification subtrees
    # Festival 1
    await verify_single_festival(
        evaluator=evaluator,
        parent_node=root,
        festival=festivals[0],
        index=0
    )

    # Festival 2
    await verify_single_festival(
        evaluator=evaluator,
        parent_node=root,
        festival=festivals[1],
        index=1
    )

    return evaluator.get_summary()