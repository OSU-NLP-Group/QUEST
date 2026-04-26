import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_dog_shows_2025"
TASK_DESCRIPTION = (
    "For each of the following three major dog shows held in the United States in 2025 - "
    "Westminster Kennel Club Dog Show, National Dog Show Presented by Purina, and AKC National Championship Presented by Royal Canin - "
    "provide the following information: (1) the specific dates the show was held, (2) the venue name and location (city and state), "
    "(3) the breed and name of the Best in Show winner, (4) the name of the handler who presented the Best in Show winner, "
    "and (5) the primary television network that broadcast the event. For each piece of information, include a reference URL from an official "
    "source or reputable news outlet that supports your answer."
)

SHOW_DISPLAY_NAMES = {
    "westminster": "Westminster Kennel Club Dog Show",
    "national": "National Dog Show Presented by Purina",
    "akc": "AKC National Championship Presented by Royal Canin"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ShowItem(BaseModel):
    show_name: Optional[str] = None  # Optional explicit name in the answer
    dates: Optional[str] = None  # e.g., "May 12–14, 2025"
    venue_name: Optional[str] = None  # e.g., "USTA Billie Jean King National Tennis Center"
    city: Optional[str] = None
    state: Optional[str] = None
    winner_breed: Optional[str] = None  # e.g., "Wire Fox Terrier"
    winner_name: Optional[str] = None  # e.g., "Buddy Holly"
    handler_name: Optional[str] = None  # e.g., "Gabriel Rangel"
    broadcast_network: Optional[str] = None  # e.g., "NBC" / "FOX" / "ABC" / "FS1"
    event_ref_urls: List[str] = Field(default_factory=list)  # URLs supporting dates/venue/location
    winner_ref_urls: List[str] = Field(default_factory=list)  # URLs supporting winner details
    broadcast_ref_urls: List[str] = Field(default_factory=list)  # URLs supporting broadcast network


class DogShowsExtraction(BaseModel):
    westminster: Optional[ShowItem] = None
    national_dog_show: Optional[ShowItem] = None
    akc_national_championship: Optional[ShowItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dog_shows() -> str:
    return """
    Extract structured information for three U.S. dog shows in 2025 from the answer:
    Shows:
      - Westminster Kennel Club Dog Show
      - National Dog Show Presented by Purina
      - AKC National Championship Presented by Royal Canin

    For each show, extract the following fields exactly as stated in the answer:
      - show_name: The show name as mentioned (or null if not explicitly given).
      - dates: The specific dates the show was held in 2025 (e.g., "May 12–14, 2025").
      - venue_name: The venue name (e.g., arena, convention center).
      - city: City of the venue.
      - state: State of the venue (use the state abbreviation or full state name as presented).
      - winner_breed: The breed of the Best in Show winner.
      - winner_name: The registered/call name of the Best in Show winner.
      - handler_name: The handler (or presenter) associated with the Best in Show winner.
      - broadcast_network: The primary television network brand that broadcast the event (e.g., NBC, FOX, ABC). If a sub-channel (e.g., FS1, NBC Sports) is mentioned, extract the exact text.
      - event_ref_urls: All URLs in the answer that support event details (dates/venue/location). Include only explicitly listed URLs.
      - winner_ref_urls: All URLs in the answer that support winner details (breed/name/handler). Include only explicitly listed URLs.
      - broadcast_ref_urls: All URLs in the answer that support the broadcast network information. Include only explicitly listed URLs.

    Important rules:
      - Only extract information explicitly present in the answer text. Do not invent or infer.
      - For each URLs field, include all valid URLs appearing in the answer (plain URLs or markdown links). If none are present, return an empty array.
      - If a required field is not present in the answer, set it to null.
      - Keep strings as they appear (allow ranges, abbreviations, capitalization, hyphens, etc.).

    Return a JSON object with fields:
      - westminster: ShowItem
      - national_dog_show: ShowItem
      - akc_national_championship: ShowItem
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_valid_url(urls: List[str]) -> bool:
    """Check if at least one URL looks valid."""
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


def combine_sources(*sources_lists: List[List[str]]) -> List[str]:
    """Combine and de-duplicate multiple URL lists while preserving order."""
    seen = set()
    combined: List[str] = []
    for sl in sources_lists:
        for u in sl:
            key = u.strip()
            if key and key not in seen:
                seen.add(key)
                combined.append(key)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_event_details_subtree(
    evaluator: Evaluator,
    parent_node,
    show_key: str,
    display_name: str,
    item: Optional[ShowItem]
) -> None:
    """
    Build the 'event details' subtree: dates, venue/location, and presence of reference URLs.
    """
    node = evaluator.add_parallel(
        id=f"{show_key}_event_details",
        desc=f"Accurate event details for {display_name} 2025",
        parent=parent_node,
        critical=False
    )

    # Critical: reference URLs must exist
    urls = item.event_ref_urls if item else []
    has_refs = has_valid_url(urls)
    evaluator.add_custom_node(
        result=has_refs,
        id=f"{show_key}_reference_url",
        desc=f"Valid reference URL supporting {display_name} event details",
        parent=node,
        critical=True
    )

    # Dates verification (critical)
    dates_leaf = evaluator.add_leaf(
        id=f"{show_key}_dates",
        desc=f"Correct dates when the {display_name} show was held in 2025",
        parent=node,
        critical=True
    )
    dates_str = item.dates if item and item.dates else ""
    await evaluator.verify(
        claim=f"The {display_name} was held on {dates_str} in 2025.",
        node=dates_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the referenced page explicitly states the event dates in 2025 as given. "
            "Allow reasonable date formatting variants and ranges (e.g., May 12–14 vs. May 12-14)."
        )
    )

    # Venue + location verification (critical)
    venue_leaf = evaluator.add_leaf(
        id=f"{show_key}_venue",
        desc=f"Correct venue name and location (city/state) for {display_name}",
        parent=node,
        critical=True
    )
    venue_name = item.venue_name if item and item.venue_name else ""
    city = item.city if item and item.city else ""
    state = item.state if item and item.state else ""
    await evaluator.verify(
        claim=f"The {display_name} took place at {venue_name} in {city}, {state}.",
        node=venue_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the referenced page explicitly lists the venue name and its location (city and state). "
            "Minor stylistic differences (e.g., abbreviations) are acceptable."
        )
    )


async def build_winner_subtree(
    evaluator: Evaluator,
    parent_node,
    show_key: str,
    display_name: str,
    item: Optional[ShowItem]
) -> None:
    """
    Build the 'winner' subtree: breed, name, handler, and presence of winner reference URLs.
    """
    node = evaluator.add_parallel(
        id=f"{show_key}_winner",
        desc=f"Accurate Best in Show winner information for {display_name} 2025",
        parent=parent_node,
        critical=False
    )

    # Critical: winner reference URLs must exist
    winner_urls = item.winner_ref_urls if item else []
    winner_has_refs = has_valid_url(winner_urls)
    evaluator.add_custom_node(
        result=winner_has_refs,
        id=f"{show_key}_winner_reference_url",
        desc=f"Valid reference URL supporting {display_name} winner information",
        parent=node,
        critical=True
    )

    # Combine winner and event URLs to strengthen verification
    all_winner_sources = combine_sources(winner_urls, item.event_ref_urls if item else [])

    # Breed verification (critical)
    breed_leaf = evaluator.add_leaf(
        id=f"{show_key}_winner_breed",
        desc=f"Correct breed of the Best in Show winner at {display_name}",
        parent=node,
        critical=True
    )
    breed = item.winner_breed if item and item.winner_breed else ""
    await evaluator.verify(
        claim=f"The Best in Show winner's breed at the 2025 {display_name} was {breed}.",
        node=breed_leaf,
        sources=all_winner_sources,
        additional_instruction=(
            "Verify the breed listed for the Best in Show winner. "
            "Accept reasonable naming variations (e.g., American vs. British breed name variants)."
        )
    )

    # Name verification (critical)
    name_leaf = evaluator.add_leaf(
        id=f"{show_key}_winner_name",
        desc=f"Correct name of the Best in Show winner at {display_name}",
        parent=node,
        critical=True
    )
    dog_name = item.winner_name if item and item.winner_name else ""
    await evaluator.verify(
        claim=f"The Best in Show winner at the 2025 {display_name} was named '{dog_name}'.",
        node=name_leaf,
        sources=all_winner_sources,
        additional_instruction=(
            "Verify the dog's registered/call name for the Best in Show winner. "
            "Allow minor punctuation/capitalization differences."
        )
    )

    # Handler verification (critical)
    handler_leaf = evaluator.add_leaf(
        id=f"{show_key}_handler",
        desc=f"Correct handler name for the Best in Show winner at {display_name}",
        parent=node,
        critical=True
    )
    handler = item.handler_name if item and item.handler_name else ""
    await evaluator.verify(
        claim=f"The Best in Show winner at the 2025 {display_name} was presented by handler {handler}.",
        node=handler_leaf,
        sources=all_winner_sources,
        additional_instruction=(
            "Verify the handler (or presenter) name associated with the Best in Show winner. "
            "If multiple co-handlers are listed, the presence of the named handler is sufficient."
        )
    )


async def build_broadcast_subtree(
    evaluator: Evaluator,
    parent_node,
    show_key: str,
    display_name: str,
    item: Optional[ShowItem]
) -> None:
    """
    Build the 'broadcast' subtree: primary television network and presence of broadcast reference URLs.
    """
    node = evaluator.add_parallel(
        id=f"{show_key}_broadcast",
        desc=f"Correct primary television network that broadcast {display_name} 2025",
        parent=parent_node,
        critical=False
    )

    # Critical: broadcast reference URLs must exist
    broadcast_urls = item.broadcast_ref_urls if item else []
    broadcast_has_refs = has_valid_url(broadcast_urls)
    evaluator.add_custom_node(
        result=broadcast_has_refs,
        id=f"{show_key}_broadcast_reference_url",
        desc=f"Valid reference URL supporting {display_name} broadcasting information",
        parent=node,
        critical=True
    )

    # Network verification (critical)
    network_leaf = evaluator.add_leaf(
        id=f"{show_key}_network",
        desc=f"Correct broadcasting network for {display_name}",
        parent=node,
        critical=True
    )
    network = item.broadcast_network if item and item.broadcast_network else ""
    await evaluator.verify(
        claim=f"The primary television network that broadcast the 2025 {display_name} was {network}.",
        node=network_leaf,
        sources=broadcast_urls,
        additional_instruction=(
            "Confirm the main broadcast network brand (e.g., NBC, FOX, ABC). "
            "Allow sub-channel equivalence where appropriate (e.g., FS1 counts under FOX; NBC Sports under NBC) "
            "if the referenced page presents it as the primary broadcast."
        )
    )


async def build_show_subtree(
    evaluator: Evaluator,
    parent_node,
    show_key: str,
    display_name: str,
    item: Optional[ShowItem]
) -> None:
    """
    Build the full subtree for a single show:
      - event details
      - winner details
      - broadcast details
    """
    show_node = evaluator.add_parallel(
        id=f"{show_key}_show",
        desc=f"Complete and accurate information for {display_name}",
        parent=parent_node,
        critical=False
    )

    # Event details
    await build_event_details_subtree(evaluator, show_node, show_key, display_name, item)

    # Winner details
    await build_winner_subtree(evaluator, show_node, show_key, display_name, item)

    # Broadcast details
    await build_broadcast_subtree(evaluator, show_node, show_key, display_name, item)


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
    Evaluate an answer for the 2025 U.S. major dog shows task.
    """
    # Initialize evaluator with a parallel root (three shows are independent)
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

    # Extract structured show info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_dog_shows(),
        template_class=DogShowsExtraction,
        extraction_name="dog_shows_2025_extraction"
    )

    # Build subtrees for each show (always create nodes; they will handle missing info via verification)
    await build_show_subtree(
        evaluator=evaluator,
        parent_node=root,
        show_key="westminster",
        display_name=SHOW_DISPLAY_NAMES["westminster"],
        item=extraction.westminster
    )

    await build_show_subtree(
        evaluator=evaluator,
        parent_node=root,
        show_key="national",
        display_name=SHOW_DISPLAY_NAMES["national"],
        item=extraction.national_dog_show
    )

    await build_show_subtree(
        evaluator=evaluator,
        parent_node=root,
        show_key="akc",
        display_name=SHOW_DISPLAY_NAMES["akc"],
        item=extraction.akc_national_championship
    )

    # Return structured summary
    return evaluator.get_summary()