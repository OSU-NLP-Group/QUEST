import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "wwe_ple_2026_first_us"
TASK_DESCRIPTION = (
    "Identify the first WWE premium live event (PLE) in calendar year 2026 that is held in the United States. "
    "For this event, provide the following information: (1) The name of the event, (2) The complete date of the event "
    "(including day of week, month, day, and year), (3) The official name of the venue where the event will be held, "
    "(4) The city where the venue is located, (5) The state where the venue is located, and (6) A URL from an official "
    "source (such as WWE.com or Ticketmaster.com) that confirms the event details. Note: Premium live events (PLEs) are "
    "major events distinct from weekly television shows like Monday Night Raw or Friday Night SmackDown."
)


class EventDetails(BaseModel):
    event_name: Optional[str] = None
    event_date_full: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


def prompt_extract_event_details() -> str:
    return """
    Extract the event details for the WWE premium live event (PLE) that the answer claims is the first PLE in the United States in calendar year 2026.

    Return a JSON object with the following fields:
    1. event_name: The official name of the event (e.g., "Royal Rumble", "WrestleMania", etc.) as stated in the answer.
    2. event_date_full: The complete date string for the event as presented in the answer, ideally including the day of week, month, day, and year (e.g., "Saturday, January 31, 2026"). If the answer uses a different but equivalent format (e.g., "Jan 31, 2026 (Sat)"), extract that string verbatim.
    3. venue_name: The official venue name where the event will be held (e.g., "T-Mobile Center", "Madison Square Garden").
    4. city: The city where the venue is located.
    5. state: The U.S. state where the venue is located (can be full name or USPS abbreviation, e.g., "PA" or "Pennsylvania").
    6. official_urls: An array of URLs explicitly provided in the answer that confirm the event details. Include only URLs explicitly present in the answer. Prefer official sources such as WWE.com or Ticketmaster.com when available. If the answer provides multiple URLs, include all of them. If the URL is missing a protocol, prepend http://.

    Important rules:
    - Extract only information explicitly mentioned in the answer text; do not invent or infer details.
    - If any field is missing in the answer, set it to null (for strings) or an empty list (for official_urls).
    - For official_urls, include exactly the URLs shown in the answer (plain or markdown links). Do not add any extra URLs not in the answer.
    """


def is_official_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower()
    return ("wwe.com" in u) or ("ticketmaster.com" in u)


async def build_and_verify_tree(evaluator: Evaluator, details: EventDetails) -> None:
    root = evaluator.root
    # Make root sequential to reflect dependency: identify first, then verify details
    # Root is already created by initialize; strategy provided there.

    # 1) Identify_Event leaf
    identify_leaf = evaluator.add_leaf(
        id="Identify_Event",
        desc="Response identifies (by name) the first WWE PLE in calendar year 2026 that is held in the United States.",
        parent=root,
        critical=True,
    )
    identified_event_name = details.event_name or ""
    identify_claim = (
        f"The first WWE premium live event (PLE) held in the United States in calendar year 2026 is '{identified_event_name}'."
    )
    await evaluator.verify(
        claim=identify_claim,
        node=identify_leaf,
        sources=details.official_urls,
        additional_instruction=(
            "Confirm via the provided official page(s) whether the named event is a WWE Premium Live Event (PLE), "
            "that it takes place in 2026, and that its location is in the United States. "
            "If schedule or event listing pages are included, ensure this is the earliest U.S.-held WWE PLE in 2026. "
            "Explicitly exclude weekly TV shows (Monday Night Raw, Friday Night SmackDown) from consideration."
        ),
    )

    # 2) Provide_Event_Details (parallel, critical)
    details_parent = evaluator.add_parallel(
        id="Provide_Event_Details",
        desc="Provide the required details for the identified event.",
        parent=root,
        critical=True,
    )

    # 2.1) Official_Source_URL (critical check presence/official domain)
    has_official = any(is_official_url(u) for u in details.official_urls)
    evaluator.add_custom_node(
        result=has_official,
        id="Official_Source_URL",
        desc="Provide a URL from an official source (e.g., WWE.com or Ticketmaster.com) that confirms the event details.",
        parent=details_parent,
        critical=True,
    )

    # 2.2) Event_Date leaf
    date_leaf = evaluator.add_leaf(
        id="Event_Date",
        desc="Provide the complete event date including day of week, month, day, and year.",
        parent=details_parent,
        critical=True,
    )
    date_str = details.event_date_full or ""
    date_claim = f"The event takes place on {date_str}."
    # 2.3) Venue_Name leaf
    venue_leaf = evaluator.add_leaf(
        id="Venue_Name",
        desc="Provide the official name of the venue where the event will be held.",
        parent=details_parent,
        critical=True,
    )
    venue_str = details.venue_name or ""
    venue_claim = f"The event will be held at '{venue_str}'."
    # 2.4) City leaf
    city_leaf = evaluator.add_leaf(
        id="City",
        desc="Provide the city where the venue is located.",
        parent=details_parent,
        critical=True,
    )
    city_str = details.city or ""
    city_claim = f"The venue city is '{city_str}'."
    # 2.5) State leaf
    state_leaf = evaluator.add_leaf(
        id="State",
        desc="Provide the state where the venue is located.",
        parent=details_parent,
        critical=True,
    )
    state_str = details.state or ""
    state_claim = f"The venue state is '{state_str}'."

    await evaluator.batch_verify(
        [
            (
                date_claim,
                details.official_urls,
                date_leaf,
                "Verify the event date on the official source page(s). Accept minor formatting differences "
                "(e.g., 'Sat, Jan 31, 2026' vs 'Saturday, January 31, 2026'). The month, day, and year must match exactly. "
                "If the official page provides the day of week, ensure consistency."
            ),
            (
                venue_claim,
                details.official_urls,
                venue_leaf,
                "Verify the official venue name on the event/ticket page. Focus on the exact venue naming as displayed on "
                "WWE.com or Ticketmaster.com. Minor variations (e.g., 'Arena' vs 'Center' if ambiguous) should be treated as incorrect."
            ),
            (
                city_claim,
                details.official_urls,
                city_leaf,
                "Verify the venue city on the official page(s). If the page shows 'City, ST' (e.g., 'Philadelphia, PA'), "
                "the city is the first component. Accept common formatting variations."
            ),
            (
                state_claim,
                details.official_urls,
                state_leaf,
                "Verify the state on the official page(s). Treat USPS abbreviations (e.g., 'PA') as equivalent to full state "
                "names ('Pennsylvania'). The state must correspond to the venue's location."
            ),
        ]
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
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
        default_model=model,
    )

    extracted_details = await evaluator.extract(
        prompt=prompt_extract_event_details(),
        template_class=EventDetails,
        extraction_name="event_details",
    )

    await build_and_verify_tree(evaluator, extracted_details)

    evaluator.add_custom_info(
        {"task_id": TASK_ID, "note": "WWE PLE 2026 first US event evaluation"},
        info_type="meta",
        info_name="evaluation_meta",
    )

    return evaluator.get_summary()