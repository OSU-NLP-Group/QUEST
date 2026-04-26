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
TASK_ID = "matteo_bocelli_dec2_2025_ca_venue"
TASK_DESCRIPTION = """
Matteo Bocelli, the son of renowned Italian tenor Andrea Bocelli, is touring North America with his 'Falling in Love World Tour' in late 2025. I'm looking to attend one of his December 2025 concerts in California at a venue that can accommodate a large audience. Specifically, I need to find a performing arts center or theater in California where he is performing on December 2, 2025, and that has a seating capacity of at least 1,700 seats. What is the name of this venue, and in which California city is it located?
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventExtraction(BaseModel):
    """
    Extracted fields from the agent's answer for the specified Matteo Bocelli Dec 2, 2025 California concert.
    """
    venue_name: Optional[str] = None  # exact venue name as listed on sources
    city: Optional[str] = None        # city within California
    state: Optional[str] = None       # expected to be CA or California if provided
    performer: Optional[str] = None   # expected "Matteo Bocelli"
    tour_name: Optional[str] = None   # expected "Falling in Love World Tour" (allowing minor variants)
    concert_date: Optional[str] = None  # expected "December 2, 2025" (allow formats like "Dec 2, 2025")
    venue_type: Optional[str] = None  # e.g., "Performing Arts Center", "Theatre", "Theater", "Concert Hall"
    capacity: Optional[str] = None    # free-form string (e.g. "2,979", "~2,800", "at least 1,700")

    # URLs explicitly cited in the answer:
    event_source_urls: List[str] = Field(default_factory=list)     # tour/venue/ticket pages listing date/venue/city
    capacity_source_urls: List[str] = Field(default_factory=list)  # venue or reliable page stating seating capacity


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_event_info() -> str:
    return """
    Extract the specific event information the answer provides for Matteo Bocelli's California concert on December 2, 2025.
    If the answer mentions multiple dates, focus only on the December 2, 2025 California show.

    Extract the following fields exactly as they appear in the answer:
    - venue_name: The exact venue name as listed on the referenced tour/venue/ticket sources (use the precise wording; do not paraphrase).
    - city: The California city for this venue (e.g., Los Angeles, San Jose, etc.).
    - state: The state, if mentioned (e.g., CA, California).
    - performer: The performer’s name (expected “Matteo Bocelli”).
    - tour_name: The tour name (expected “Falling in Love World Tour”; if the answer uses a close variant, extract that).
    - concert_date: The date for the concert (expected “December 2, 2025”; allow variants like “Dec 2, 2025”).
    - venue_type: If the answer explicitly characterizes the venue type (e.g., performing arts center, theater, concert hall), extract it.
    - capacity: If the answer provides a seating capacity number or range, extract the text exactly (e.g., “2,979 seats”, “~2,800”, “over 1,700”).
    - event_source_urls: All URLs cited in the answer that specifically support the event details (date/venue/city/tour/perfomer).
    - capacity_source_urls: All URLs cited in the answer that specifically support the venue seating capacity.

    Notes:
    - Only extract URLs that are explicitly present in the answer (including markdown links).
    - Do not invent URLs.
    - If a field is not present in the answer, return null (or an empty list for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(*url_lists: List[str]) -> List[str]:
    """Combine multiple URL lists into a unique, ordered list, skipping empties."""
    seen = set()
    combined: List[str] = []
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if u and (u not in seen):
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_answer_output_checks(evaluator: Evaluator, parent, extracted: EventExtraction) -> None:
    """
    Answer_Output (critical) group:
      - Venue_Name_Provided (existence only; split out for clarity)
      - Venue_Name_As_Listed (source-supported exact listing wording)
      - City_Provided (existence only)
    """
    answer_output = evaluator.add_parallel(
        id="Answer_Output",
        desc="Provide the requested output fields (venue name and city).",
        parent=parent,
        critical=True
    )

    # Existence: venue name provided
    evaluator.add_custom_node(
        result=bool(extracted.venue_name and extracted.venue_name.strip()),
        id="Venue_Name_Provided",
        desc="Venue name is provided in the answer.",
        parent=answer_output,
        critical=True
    )

    # Accuracy vs sources: exact as listed (case-insensitive, allow minor punctuation variants)
    venue_name_leaf = evaluator.add_leaf(
        id="Venue_Name_As_Listed",
        desc="Venue name is provided and matches the official tour listing wording (exact name as listed).",
        parent=answer_output,
        critical=True
    )
    name_to_check = extracted.venue_name or ""
    event_urls = extracted.event_source_urls

    name_claim = (
        f"For the California concert on December 2, 2025, the venue name shown on the cited tour/venue/ticket source(s) "
        f"is exactly '{name_to_check}' (treating case-insensitive and allowing minor punctuation normalization)."
    )
    await evaluator.verify(
        claim=name_claim,
        node=venue_name_leaf,
        sources=event_urls,
        additional_instruction=(
            "Find the listing for the Dec 2, 2025 California show on the provided page(s) and compare the venue "
            "name to the answer's venue_name exactly as written (case-insensitive; allow minor punctuation variants like apostrophes or hyphens). "
            "Prefer the official tour site or the venue’s official page if available; reliable ticketing sites are acceptable too."
        )
    )

    # Existence: city provided
    evaluator.add_custom_node(
        result=bool(extracted.city and extracted.city.strip()),
        id="City_Provided",
        desc="A specific city is provided for the venue location.",
        parent=answer_output,
        critical=True
    )


async def build_event_constraints_checks(evaluator: Evaluator, parent, extracted: EventExtraction) -> None:
    """
    Event_Constraints_Verification (critical) group:
      - Performer_Is_Matteo_Bocelli_Son_of_Andrea (verify performer is Matteo Bocelli)
      - Tour_Name_Matches (verify falling in love world tour)
      - Concert_Date_Matches (verify date is Dec 2, 2025)
    """
    event_group = evaluator.add_parallel(
        id="Event_Constraints_Verification",
        desc="The identified event matches the specified performer, tour, and date constraints.",
        parent=parent,
        critical=True
    )

    # Performer check
    performer_leaf = evaluator.add_leaf(
        id="Performer_Is_Matteo_Bocelli_Son_of_Andrea",
        desc="Performer is Matteo Bocelli (son of Andrea Bocelli).",
        parent=event_group,
        critical=True
    )
    await evaluator.verify(
        claim="The performer for this event is Matteo Bocelli.",
        node=performer_leaf,
        sources=extracted.event_source_urls,
        additional_instruction=(
            "Only verify that the event is for Matteo Bocelli. The phrase 'son of Andrea Bocelli' is contextual and "
            "does not need to be explicitly present on the page."
        )
    )

    # Tour name check
    tour_leaf = evaluator.add_leaf(
        id="Tour_Name_Matches",
        desc="Tour name is “Falling in Love World Tour.”",
        parent=event_group,
        critical=True
    )
    tour_name_to_expect = "Falling in Love World Tour"
    tour_claim = (
        f"This event is part of the '{tour_name_to_expect}' (allow minor variants like 'Falling in Love Tour' or "
        f"'Falling In Love World Tour')."
    )
    await evaluator.verify(
        claim=tour_claim,
        node=tour_leaf,
        sources=extracted.event_source_urls,
        additional_instruction=(
            "Confirm the tour branding on the event/tour page(s). Accept close variants such as capitalization differences "
            "or omission of the word 'World' if the context clearly indicates the same tour."
        )
    )

    # Date check
    date_leaf = evaluator.add_leaf(
        id="Concert_Date_Matches",
        desc="Concert date is December 2, 2025.",
        parent=event_group,
        critical=True
    )
    date_claim = "The event date for this show is December 2, 2025 (accept 'Dec 2, 2025')."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=extracted.event_source_urls,
        additional_instruction=(
            "Find the entry corresponding to the California engagement and verify that the listed date is December 2, 2025. "
            "Formatting variations like 'Dec 2, 2025' are acceptable."
        )
    )


async def build_venue_constraints_checks(evaluator: Evaluator, parent, extracted: EventExtraction) -> None:
    """
    Venue_Constraints_Verification (critical) group:
      - Venue_In_California_USA
      - Venue_Is_Performing_Arts_Center_Or_Theater
      - Venue_Capacity_At_Least_1700
    """
    venue_group = evaluator.add_parallel(
        id="Venue_Constraints_Verification",
        desc="The identified venue meets the location, type, and capacity constraints.",
        parent=parent,
        critical=True
    )

    # Location: California, USA
    ca_leaf = evaluator.add_leaf(
        id="Venue_In_California_USA",
        desc="Venue is located in California, United States.",
        parent=venue_group,
        critical=True
    )
    city_txt = extracted.city or "the specified city"
    venue_name_txt = extracted.venue_name or "the specified venue"
    ca_claim = (
        f"{venue_name_txt} is located in {city_txt}, California, United States. "
        f"(Accept 'CA' as equivalent to 'California'.)"
    )
    await evaluator.verify(
        claim=ca_claim,
        node=ca_leaf,
        sources=extracted.event_source_urls,
        additional_instruction=(
            "Use the provided source(s) to confirm that the venue for the Dec 2, 2025 show is in California, USA. "
            "Abbreviations like 'CA' should be treated as 'California'."
        )
    )

    # Type: performing arts center or theater
    type_leaf = evaluator.add_leaf(
        id="Venue_Is_Performing_Arts_Center_Or_Theater",
        desc="Venue is a dedicated performing arts center or theater facility.",
        parent=venue_group,
        critical=True
    )
    type_sources = _unique_urls(extracted.event_source_urls, extracted.capacity_source_urls)
    type_claim = (
        f"{venue_name_txt} is a performing arts center or theater (concert hall/auditorium that functions as a theatre counts)."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=type_sources,
        additional_instruction=(
            "Confirm that the venue is a dedicated performance facility, such as a performing arts center, theatre/theater, "
            "concert hall, or auditorium within a performing arts complex. Terminology variants are acceptable if the intent is clear."
        )
    )

    # Capacity: at least 1,700 seats
    capacity_leaf = evaluator.add_leaf(
        id="Venue_Capacity_At_Least_1700",
        desc="Venue seating capacity is at least 1,700 seats.",
        parent=venue_group,
        critical=True
    )
    cap_sources = extracted.capacity_source_urls if extracted.capacity_source_urls else extracted.event_source_urls
    capacity_claim = (
        f"The seating capacity of {venue_name_txt} is at least 1,700 seats."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=cap_sources,
        additional_instruction=(
            "Look for stated seating capacity on the venue’s official site or a reliable source (e.g., Wikipedia, major publications). "
            "If multiple halls are mentioned, focus on the hall used for the concert; if unclear, consider the main theatre capacity. "
            "Ranges or approximations that clearly indicate 1,700 or more are acceptable."
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
    Evaluate an answer for the Matteo Bocelli Dec 2, 2025 California venue task.
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted: EventExtraction = await evaluator.extract(
        prompt=prompt_extract_event_info(),
        template_class=EventExtraction,
        extraction_name="event_extraction"
    )

    # Build top-level critical node
    venue_ident_node = evaluator.add_parallel(
        id="Venue_Identification",
        desc="Identify the correct California venue and city for Matteo Bocelli’s Dec 2, 2025 concert on the Falling in Love World Tour, meeting the capacity and venue-type constraints, and report the venue name (as listed) and city.",
        parent=root,
        critical=True
    )

    # Subgroups and leaves
    await build_answer_output_checks(evaluator, venue_ident_node, extracted)
    await build_event_constraints_checks(evaluator, venue_ident_node, extracted)
    await build_venue_constraints_checks(evaluator, venue_ident_node, extracted)

    # Return summary
    return evaluator.get_summary()