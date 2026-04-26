import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "south_florida_broadway_2026"
TASK_DESCRIPTION = (
    "Find four different touring Broadway shows that will be performing in South Florida between March 1, 2026 and "
    "June 30, 2026. For this task, South Florida includes Miami, Fort Lauderdale, and West Palm Beach. Each show must "
    "meet the following criteria: The show must be a touring Broadway production (not a local production); The performance "
    "venue must be located in Miami, Fort Lauderdale, or West Palm Beach, Florida; The show must have performances scheduled "
    "between March 1, 2026 and June 30, 2026; The venue must have a seating capacity of at least 2,000 seats; The venue must "
    "offer wheelchair-accessible seating options. For each of the four shows, provide: (1) The official show name, (2) The venue "
    "name, (3) The specific performance dates during the March-June 2026 timeframe, (4) The venue's seating capacity, (5) Confirmation "
    "that the venue offers wheelchair-accessible seating, (6) A reference URL from an official source (venue website, tour website, or "
    "ticketing platform) that confirms the show details."
)

DATE_RANGE_START = "2026-03-01"
DATE_RANGE_END = "2026-06-30"
ALLOWED_CITIES = ["Miami", "Fort Lauderdale", "West Palm Beach"]

ORDINAL_DESC = {
    0: "First Broadway touring show meeting all criteria",
    1: "Second Broadway touring show meeting all criteria",
    2: "Third Broadway touring show meeting all criteria",
    3: "Fourth Broadway touring show meeting all criteria",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShowItem(BaseModel):
    show_name: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    dates: Optional[str] = None  # Keep as free-form text for robustness
    capacity: Optional[str] = None  # Free-form, e.g., "2,648" or "about 2600"
    accessibility_info: Optional[str] = None  # Free-form mention of wheelchair access
    source_urls: List[str] = Field(default_factory=list)  # All URLs the answer cites for this show
    type_text: Optional[str] = None  # Free-form notes like "national tour", "Broadway in Fort Lauderdale", etc.


class ShowsExtraction(BaseModel):
    shows: List[ShowItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shows() -> str:
    return """
    Extract up to the first FOUR shows that the answer proposes for the task. For each show, extract:

    - show_name: The show’s official title as written in the answer.
    - venue_name: The venue name where the show will be performed.
    - city: The city for the venue (e.g., "Miami", "Fort Lauderdale", or "West Palm Beach" if stated; otherwise copy the city text as given).
    - dates: The specific performance dates mentioned in the answer for the March–June 2026 timeframe. Keep as a single string exactly as written.
    - capacity: The seating capacity mentioned in the answer for the venue (if provided). Keep the text as-is (e.g., "2,648 seats").
    - accessibility_info: Any mention confirming wheelchair-accessible seating (copy the wording if present; else null).
    - type_text: Any phrase indicating this is a touring Broadway production (e.g., "Broadway in [City]", "National Tour", "Touring Broadway", "Broadway Across America", etc.). If not mentioned, set to null.
    - source_urls: An array with all URLs provided in the answer that are associated with this show (include venue pages, the tour’s official site, and ticketing platforms). If no URL is present, return an empty array.

    Do not invent or infer data. Only extract what is explicitly provided in the answer text. If a field is not mentioned, set it to null (or an empty array for source_urls).
    """


# --------------------------------------------------------------------------- #
# Verification for a single show                                              #
# --------------------------------------------------------------------------- #
async def verify_one_show(
    evaluator: Evaluator,
    parent_node,
    show: ShowItem,
    idx: int,
) -> None:
    """
    Build verification sub-tree and run checks for one show.
    """
    # Parent node for this show
    show_node = evaluator.add_parallel(
        id=f"show_{idx + 1}",
        desc=ORDINAL_DESC.get(idx, f"Show #{idx + 1} verification"),
        parent=parent_node,
        critical=False
    )

    urls = list(dict.fromkeys(show.source_urls or []))  # dedupe while preserving order

    # 8. URL existence check (critical) – ensure at least one reference URL is provided
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id=f"show_{idx + 1}_url",
        desc="A valid reference URL is provided that confirms the show details",
        parent=show_node,
        critical=True
    )

    # Prepare leaf nodes
    name_node = evaluator.add_leaf(
        id=f"show_{idx + 1}_name",
        desc="The official show name is correctly identified",
        parent=show_node,
        critical=True
    )
    type_node = evaluator.add_leaf(
        id=f"show_{idx + 1}_type",
        desc="The show is a touring Broadway production, not a local production or concert",
        parent=show_node,
        critical=True
    )
    venue_node = evaluator.add_leaf(
        id=f"show_{idx + 1}_venue",
        desc="The venue name where the show is performing is correctly identified",
        parent=show_node,
        critical=True
    )
    location_node = evaluator.add_leaf(
        id=f"show_{idx + 1}_location",
        desc="The venue is located in Miami, Fort Lauderdale, or West Palm Beach, Florida",
        parent=show_node,
        critical=True
    )
    dates_node = evaluator.add_leaf(
        id=f"show_{idx + 1}_dates",
        desc="The performance dates fall within March 1 - June 30, 2026",
        parent=show_node,
        critical=True
    )
    capacity_node = evaluator.add_leaf(
        id=f"show_{idx + 1}_capacity",
        desc="The venue has a seating capacity of at least 2,000 seats",
        parent=show_node,
        critical=True
    )
    access_node = evaluator.add_leaf(
        id=f"show_{idx + 1}_accessibility",
        desc="The venue offers wheelchair-accessible seating",
        parent=show_node,
        critical=True
    )

    # Build claims
    name_val = (show.show_name or "").strip()
    venue_val = (show.venue_name or "").strip()

    claim_name = (
        f"The show/event title shown on the provided page(s) is '{name_val}' or a close equivalent "
        f"(minor variants like subtitles, 'The Musical', or possessives are acceptable)."
    )
    claim_type = (
        "The provided page(s) indicate that this event is a touring Broadway production (e.g., a national tour, "
        "part of a 'Broadway in [City]' or 'Broadway Across America' series), and not a locally produced community or regional staging, "
        "and not a concert."
    )
    claim_venue = (
        f"The event is scheduled at the venue '{venue_val}' (minor style variations acceptable)."
    )
    claim_location = (
        "The venue’s address/location shown on the page(s) is in one of these cities in Florida: Miami, Fort Lauderdale, or West Palm Beach."
    )
    claim_dates = (
        f"The event page shows at least one performance date between {DATE_RANGE_START} and {DATE_RANGE_END} (inclusive). "
        "If the run spans a wider window, it still passes as long as at least one date falls within this range."
    )
    claim_capacity = (
        "The venue’s seating capacity (for the specific hall/theater used by this event) is at least 2,000 seats."
    )
    claim_access = (
        "The venue offers wheelchair-accessible seating options."
    )

    # Additional instructions tailored per check
    ins_name = (
        "Match the page’s event title to the answer’s show name. Allow minor formatting differences, subtitle/tagline inclusion, "
        "and common variants like 'The Musical'."
    )
    ins_type = (
        "Look for phrases like 'National Tour', 'Broadway tour', 'Touring Broadway', 'Broadway in Miami', 'Broadway Across America', "
        "'Bank of America Broadway', etc. If the page makes it clear the production is a touring Broadway engagement, pass. "
        "If it appears to be a local/regional production, fail."
    )
    ins_venue = (
        "Confirm that the page lists the specific venue where the show will perform. Minor naming variations are acceptable."
    )
    ins_location = (
        "Confirm the city is Miami, Fort Lauderdale (also acceptable as 'Ft Lauderdale'/'Ft. Lauderdale'), or West Palm Beach. "
        "Treat 'Miami Beach' as part of the Miami area for this task."
    )
    ins_dates = (
        "Check the event schedule/date list or the range of performance dates. If any date is between 2026-03-01 and 2026-06-30 (inclusive), pass. "
        "Ignore time zones; focus on dates as displayed."
    )
    ins_capacity = (
        "Use a venue information page (venue site, a venue profile page, Wikipedia if included among the provided URLs, etc.). "
        "Be careful with multi-theater complexes; ensure the specific hall used by the Broadway engagement meets or exceeds 2,000 seats."
    )
    ins_access = (
        "Look for 'Accessibility', 'ADA', 'wheelchair accessible seating', or similar language on the venue or event page. "
        "A general venue accessibility policy that includes wheelchair seating is acceptable."
    )

    # Collect all verifications to run in parallel
    claims_and_sources = [
        (claim_name, urls, name_node, ins_name),
        (claim_type, urls, type_node, ins_type),
        (claim_venue, urls, venue_node, ins_venue),
        (claim_location, urls, location_node, ins_location),
        (claim_dates, urls, dates_node, ins_dates),
        (claim_capacity, urls, capacity_node, ins_capacity),
        (claim_access, urls, access_node, ins_access),
    ]

    # Execute verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the South Florida touring Broadway shows (Mar–Jun 2026) task.
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_shows(),
        template_class=ShowsExtraction,
        extraction_name="shows_extraction",
    )

    # Use up to the first 4 shows; pad with empty shells if fewer provided
    shows: List[ShowItem] = list(extracted.shows[:4])
    while len(shows) < 4:
        shows.append(ShowItem())

    # Build subtrees for each show (parallel)
    verify_tasks = []
    for i in range(4):
        verify_tasks.append(verify_one_show(evaluator, root, shows[i], i))

    await asyncio.gather(*verify_tasks)

    # Return structured summary
    return evaluator.get_summary()