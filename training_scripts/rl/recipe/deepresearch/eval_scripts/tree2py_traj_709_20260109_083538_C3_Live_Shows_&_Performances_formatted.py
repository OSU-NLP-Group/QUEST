import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_tour_midwest_2026"
TASK_DESCRIPTION = (
    "Identify a Broadway musical that has a national touring production scheduled to perform in the United States during 2026. "
    "From this tour, find a specific engagement taking place in a Midwestern U.S. state (Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, or Wisconsin) "
    "during the first half of 2026 (January 1 through June 30, 2026). For this engagement, provide: "
    "1. The name of the Broadway musical, "
    "2. The specific venue name and city where it will perform, "
    "3. The venue's seating capacity, "
    "4. Confirmation that the venue is a theater or performing arts center with a proscenium or comparable stage (not an arena, stadium, or outdoor venue), "
    "5. The complete date range of the engagement (start and end dates), "
    "6. Where tickets can be purchased or how ticket information can be accessed, "
    "7. The venue's complete physical address (street address, city, state, and ZIP code). "
    "Include direct URLs to: (a) the tour's official website or a major theatrical tour listing website confirming the tour schedule, and "
    "(b) the venue's official website or a reliable source documenting the venue's specifications."
)

MIDWEST_STATES = {
    "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota",
    "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin",
    # Accept common abbreviations too (used in reasoning, but validation uses URLs)
    "IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShowInfo(BaseModel):
    name: Optional[str] = None
    tour_urls: List[str] = Field(default_factory=list)


class EngagementInfo(BaseModel):
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    engagement_urls: List[str] = Field(default_factory=list)
    tickets_url: Optional[str] = None  # direct ticketing URL if available


class VenueInfo(BaseModel):
    capacity: Optional[str] = None  # keep as string to allow ranges/approx
    stage_type_description: Optional[str] = None  # e.g., "proscenium theater"
    venue_urls: List[str] = Field(default_factory=list)  # official venue pages or reliable specs pages
    address_street: Optional[str] = None
    address_city: Optional[str] = None
    address_state: Optional[str] = None
    address_zip: Optional[str] = None


class TourDataExtraction(BaseModel):
    show: Optional[ShowInfo] = None
    engagement: Optional[EngagementInfo] = None
    venue: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_tour_data() -> str:
    return """
    Extract structured information from the answer about a Broadway musical tour engagement in early 2026 in the U.S. Midwest. 
    Only extract information explicitly mentioned in the answer. If multiple engagements are presented, select the first one that the answer claims is in a Midwestern state and occurs between January 1 and June 30, 2026.
    
    Return a JSON object with the following nested structure:

    {
      "show": {
        "name": string | null,
        "tour_urls": [url, ...]  // direct tour schedule URLs; official show tour site or major listings like Broadway.org, Playbill, etc.
      },
      "engagement": {
        "venue_name": string | null,
        "venue_city": string | null,
        "venue_state": string | null,
        "start_date": string | null,  // exactly as in the answer (e.g., "March 12, 2026")
        "end_date": string | null,    // closing date or end of engagement as stated
        "engagement_urls": [url, ...], // direct URLs confirming the specific engagement, e.g., show tour stop page or venue event page
        "tickets_url": string | null   // direct ticketing/purchase URL if provided; else null
      },
      "venue": {
        "capacity": string | null,               // seating capacity (number or range) as presented
        "stage_type_description": string | null, // phrases like "proscenium theater", "performing arts center with stage", etc.
        "venue_urls": [url, ...],                // official venue website or reliable source pages documenting specs
        "address_street": string | null,
        "address_city": string | null,
        "address_state": string | null,
        "address_zip": string | null
      }
    }

    Special notes:
    - URLs must be explicitly present in the answer. Accept plain URLs or markdown-formatted links; extract the actual URLs.
    - Do not invent any dates, addresses, or capacities. If the answer omits a field, return null (or [] for arrays).
    - Use strings for dates and capacities to allow flexibility and ranges.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def format_address(venue: VenueInfo) -> str:
    street = venue.address_street or ""
    city = venue.address_city or ""
    state = venue.address_state or ""
    zip_code = venue.address_zip or ""
    parts = [street.strip(), city.strip(), state.strip(), zip_code.strip()]
    # Basic join with commas, then remove redundant spaces/commas
    addr = ", ".join([p for p in parts if p])
    return addr


def combine_sources(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for ul in url_lists:
        for u in ul:
            if isinstance(u, str) and u.strip() and u not in combined:
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_show_identification(
    evaluator: Evaluator,
    parent_node,
    data: TourDataExtraction
) -> None:
    show_node = evaluator.add_sequential(
        id="show_identification",
        desc="Identify a Broadway musical that has a national touring production scheduled to perform in the United States during 2026",
        parent=parent_node,
        critical=True
    )

    show_type_and_tour = evaluator.add_parallel(
        id="show_type_and_tour",
        desc="Verify the show is a Broadway musical with confirmed 2026 U.S. tour dates",
        parent=show_node,
        critical=True
    )

    # Leaf: show_is_broadway_musical
    show_name = (data.show.name if data.show else None) or ""
    show_urls = (data.show.tour_urls if data.show else []) or []
    mus_leaf = evaluator.add_leaf(
        id="show_is_broadway_musical",
        desc="The identified show is a Broadway musical (not a play, comedy show, concert, or other performance type)",
        parent=show_type_and_tour,
        critical=True
    )
    mus_claim = f"The show '{show_name}' is a Broadway musical."
    await evaluator.verify(
        claim=mus_claim,
        node=mus_leaf,
        sources=show_urls,
        additional_instruction="Use the provided tour/official/listing URLs to confirm the show is categorized as a Broadway musical (not a straight play or other performance). Allow common naming variations."
    )

    # Leaf: show_has_2026_tour
    tour_leaf = evaluator.add_leaf(
        id="show_has_2026_tour",
        desc="The show has confirmed touring dates scheduled in the United States during the year 2026",
        parent=show_type_and_tour,
        critical=True
    )
    tour_claim = "This show has confirmed U.S. tour dates scheduled during the year 2026."
    await evaluator.verify(
        claim=tour_claim,
        node=tour_leaf,
        sources=show_urls,
        additional_instruction="Confirm the tour schedule on the provided tour/listing pages includes 2026 U.S. dates (any month in 2026)."
    )

    # Leaf (existence via custom): show_reference_url
    evaluator.add_custom_node(
        result=bool(show_urls),
        id="show_reference_url",
        desc="A direct URL to the show's official tour website or a major theatrical tour listing website (Broadway.org, Playbill, etc.) confirming the tour is provided",
        parent=show_node,
        critical=True
    )


async def build_tour_stop_selection(
    evaluator: Evaluator,
    parent_node,
    data: TourDataExtraction
) -> None:
    tour_stop_node = evaluator.add_sequential(
        id="tour_stop_selection",
        desc="Identify a specific tour stop for the identified musical that takes place in a Midwestern U.S. state during the first half of 2026",
        parent=parent_node,
        critical=True
    )

    geo_time_node = evaluator.add_parallel(
        id="geographic_and_temporal_criteria",
        desc="Verify the tour stop meets both geographic (Midwestern state) and temporal (January-June 2026) requirements",
        parent=tour_stop_node,
        critical=True
    )

    engagement = data.engagement or EngagementInfo()
    venue = data.venue or VenueInfo()
    engagement_sources = (engagement.engagement_urls or [])
    venue_sources = (venue.venue_urls or [])
    combined_engagement_sources = combine_sources(engagement_sources, venue_sources)

    # Leaf: location_in_midwest
    loc_leaf = evaluator.add_leaf(
        id="location_in_midwest",
        desc="The tour stop is located in a Midwestern U.S. state (Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, or Wisconsin)",
        parent=geo_time_node,
        critical=True
    )
    state_str = (engagement.venue_state or venue.address_state or "") or ""
    city_str = (engagement.venue_city or venue.address_city or "") or ""
    loc_claim = f"The tour stop is located in {city_str}, {state_str}, which is a Midwestern U.S. state."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=combined_engagement_sources,
        additional_instruction="Check the event/venue pages to confirm the venue's city/state. Consider the following as Midwest: Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, Wisconsin (and their postal abbreviations)."
    )

    # Leaf: dates_in_first_half_2026
    dates_leaf = evaluator.add_leaf(
        id="dates_in_first_half_2026",
        desc="The tour stop's performance dates fall within January 1, 2026 through June 30, 2026",
        parent=geo_time_node,
        critical=True
    )
    start_str = (engagement.start_date or "") or ""
    end_str = (engagement.end_date or "") or ""
    dates_claim = f"The engagement runs from {start_str} to {end_str}, and both dates fall within January 1, 2026 through June 30, 2026."
    await evaluator.verify(
        claim=dates_claim,
        node=dates_leaf,
        sources=engagement_sources,
        additional_instruction="Verify on the event/tour listing pages that the start and end/closing dates are both between Jan 1 and Jun 30, 2026. Allow minor formatting variations in date naming."
    )

    # Leaf (existence via custom): specific_venue_identified
    evaluator.add_custom_node(
        result=bool(engagement.venue_name and ((engagement.venue_city or venue.address_city))),
        id="specific_venue_identified",
        desc="The specific venue name and city where the tour stop takes place is provided",
        parent=tour_stop_node,
        critical=True
    )


async def build_venue_verification(
    evaluator: Evaluator,
    parent_node,
    data: TourDataExtraction
) -> None:
    venue_node = evaluator.add_sequential(
        id="venue_verification",
        desc="Verify that the identified venue meets the required technical specifications suitable for hosting Broadway touring productions",
        parent=parent_node,
        critical=True
    )

    specs_node = evaluator.add_parallel(
        id="venue_technical_specs",
        desc="Document the venue's seating capacity and verify it is an appropriate theater type with proscenium or comparable stage",
        parent=venue_node,
        critical=True
    )

    venue = data.venue or VenueInfo()
    v_urls = venue.venue_urls or []

    # Leaf: venue_seating_capacity
    cap_leaf = evaluator.add_leaf(
        id="venue_seating_capacity",
        desc="The venue's seating capacity is documented and provided (must be verifiable from venue's official website or reliable source)",
        parent=specs_node,
        critical=True
    )
    cap_claim = f"The venue's seating capacity is {venue.capacity}."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=v_urls,
        additional_instruction="Confirm capacity from official venue pages or reliable documentation. Allow approximate ranges (e.g., 'about 2,500')."
    )

    # Leaf: venue_type_appropriate
    type_leaf = evaluator.add_leaf(
        id="venue_type_appropriate",
        desc="The venue is a theater or performing arts center with a proscenium or comparable stage (not an arena, stadium, or outdoor venue)",
        parent=specs_node,
        critical=True
    )
    type_desc = venue.stage_type_description or ""
    type_claim = (
        "The venue is an appropriate theater or performing arts center with a proscenium or comparable stage, "
        "and it is not an arena, stadium, or outdoor venue."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=v_urls,
        additional_instruction="Use venue pages to confirm it's a theater/performing arts center and suitable for Broadway touring productions. Mentions of 'proscenium', 'theatre', 'performing arts center', or similar architectural/stage descriptions are supporting evidence."
    )

    # Leaf (existence via custom): venue_reference_url
    evaluator.add_custom_node(
        result=bool(v_urls),
        id="venue_reference_url",
        desc="A direct URL to the venue's official website or a reliable source documenting the venue's specifications is provided",
        parent=venue_node,
        critical=True
    )


async def build_engagement_information(
    evaluator: Evaluator,
    parent_node,
    data: TourDataExtraction
) -> None:
    engage_node = evaluator.add_parallel(
        id="engagement_information",
        desc="Provide complete details about the specific tour engagement including performance dates and ticket purchase information",
        parent=parent_node,
        critical=True
    )

    engagement = data.engagement or EngagementInfo()

    # Leaf: performance_date_range
    perf_leaf = evaluator.add_leaf(
        id="performance_date_range",
        desc="The complete date range of the engagement (start date and end date or closing date) is provided",
        parent=engage_node,
        critical=True
    )
    perf_claim = f"The engagement runs from {engagement.start_date} to {engagement.end_date}."
    await evaluator.verify(
        claim=perf_claim,
        node=perf_leaf,
        sources=engagement.engagement_urls,
        additional_instruction="Confirm both start and end/closing dates on the engagement/tour pages. Accept minor formatting variations for dates."
    )

    # Leaf: ticket_information_source
    ticket_leaf = evaluator.add_leaf(
        id="ticket_information_source",
        desc="A reference to where tickets can be purchased or information about ticket availability is provided (venue box office, Ticketmaster, Broadway tour website, etc.)",
        parent=engage_node,
        critical=True
    )
    t_url = engagement.tickets_url or ""
    ticket_claim = (
        f"Tickets can be purchased or ticket information is available at the provided source: {t_url if t_url else 'one of the engagement pages'}."
    )
    ticket_sources = [engagement.tickets_url] if engagement.tickets_url else engagement.engagement_urls
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_leaf,
        sources=ticket_sources,
        additional_instruction="Verify that the provided page(s) contain ticket purchase links or clear ticket availability information for the specific engagement."
    )


async def build_venue_address_leaf(
    evaluator: Evaluator,
    parent_node,
    data: TourDataExtraction
) -> None:
    # Single leaf under root per rubric
    venue = data.venue or VenueInfo()
    addr_leaf = evaluator.add_leaf(
        id="venue_address",
        desc="Provide the venue's complete physical address including street address, city, state, and ZIP code",
        parent=parent_node,
        critical=True
    )
    address_text = format_address(venue)
    addr_claim = f"The venue's complete physical address is: '{address_text}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=venue.venue_urls,
        additional_instruction="Confirm the full street address, city, state, and ZIP on official venue pages (e.g., Contact, Visit, About). Minor formatting differences are acceptable."
    )


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for identifying a qualifying Broadway touring musical with a specific Midwest engagement in early 2026.
    Builds a sequential critical verification tree mirroring the rubric and returns a standard evaluation summary.
    """
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
        default_model=model
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_tour_data(),
        template_class=TourDataExtraction,
        extraction_name="tour_data_extraction"
    )

    # Build tree following rubric
    await build_show_identification(evaluator, root, extracted)
    await build_tour_stop_selection(evaluator, root, extracted)
    await build_venue_verification(evaluator, root, extracted)
    await build_engagement_information(evaluator, root, extracted)
    await build_venue_address_leaf(evaluator, root, extracted)

    # Return the aggregated summary
    return evaluator.get_summary()