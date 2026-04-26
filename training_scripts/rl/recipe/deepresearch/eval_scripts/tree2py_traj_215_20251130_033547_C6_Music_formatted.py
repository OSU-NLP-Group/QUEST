import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grammys_gnr_chain"
TASK_DESCRIPTION = (
    "At the 2024 Grammy Awards, an album won the Album of the Year category. Identify the main producer or co-producer "
    "of this album who also won Producer of the Year (Non-Classical) at the same ceremony. Determine the U.S. state "
    "where this producer was born or raised. On the Guns N' Roses 2026 North American tour, find the venue located in "
    "this state. For this venue, provide: (1) the exact date of the concert (month, day, and year), (2) the seating "
    "capacity of the venue for concerts, and (3) the city and state where the venue is located. Include reference URLs "
    "for all findings."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class AlbumInfo(BaseModel):
    """Album of the Year winner information and supporting sources."""
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ProducerInfo(BaseModel):
    """Producer identity, role on the album, and Producer of the Year award info."""
    name: Optional[str] = None
    role_type: Optional[str] = None  # e.g., "main producer" or "co-producer"
    role_sources: List[str] = Field(default_factory=list)
    award_sources: List[str] = Field(default_factory=list)


class ProducerStateInfo(BaseModel):
    """Producer's U.S. birth or raised state and sources."""
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    """Guns N' Roses 2026 North American tour venue in the producer's state."""
    name: Optional[str] = None
    tour_sources: List[str] = Field(default_factory=list)


class VenueDetails(BaseModel):
    """Detailed venue information and supporting sources."""
    concert_date: Optional[str] = None
    date_sources: List[str] = Field(default_factory=list)
    capacity: Optional[str] = None  # Keep as string to support ranges/narratives
    capacity_sources: List[str] = Field(default_factory=list)
    city: Optional[str] = None
    state: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)


class ResearchChainExtraction(BaseModel):
    """Top-level extracted information for the entire research chain."""
    album: Optional[AlbumInfo] = None
    producer: Optional[ProducerInfo] = None
    producer_state: Optional[ProducerStateInfo] = None
    venue: Optional[VenueInfo] = None
    venue_details: Optional[VenueDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chain() -> str:
    return """
    Extract the structured information from the answer for the following research chain. You must only extract exactly what is stated in the answer. If something is missing, return null or an empty list.

    Required JSON structure:
    {
      "album": {
        "name": string | null,
        "sources": string[]  // URLs that support that this album won 2024 Grammys Album of the Year
      },
      "producer": {
        "name": string | null,
        "role_type": string | null,   // for example: "main producer" or "co-producer"
        "role_sources": string[],     // URLs that support the producer is main/co-producer of the album
        "award_sources": string[]     // URLs that support the producer won Producer of the Year (Non-Classical) at the 2024 Grammys
      },
      "producer_state": {
        "state": string | null,       // the U.S. state where the producer was born or raised (either is acceptable per the task)
        "sources": string[]           // URLs that support this state claim
      },
      "venue": {
        "name": string | null,        // the venue on the Guns N' Roses 2026 North American tour located in the producer's state
        "tour_sources": string[]      // URLs that support that the identified venue is indeed on the 2026 North American tour (and ideally indicates location)
      },
      "venue_details": {
        "concert_date": string | null,        // exact date (month, day, year) of the concert at this venue on GNR 2026 tour
        "date_sources": string[],             // URLs supporting the concert date
        "capacity": string | null,            // seating capacity for concerts at the venue
        "capacity_sources": string[],         // URLs supporting the concert capacity (for concerts)
        "city": string | null,                // city of the venue
        "state": string | null,               // state of the venue
        "location_sources": string[]          // URLs supporting the venue's city and state location
      }
    }

    Important instructions:
    - Extract only URLs explicitly present in the answer. If the answer references sources but does not include URLs, return an empty list.
    - Accept URLs in plain form or markdown link format. Always extract the underlying URL.
    - If any subfield is not mentioned in the answer, set it to null (or [] for sources).
    - Do not infer or fabricate any values. Keep capacity as a string (it may include ranges or qualifiers like "concert configuration").
    - If multiple venues are mentioned, choose the one explicitly tied to the producer's state.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_list(x: Optional[List[str]]) -> List[str]:
    return x if isinstance(x, list) else []


# --------------------------------------------------------------------------- #
# Step verifications                                                          #
# --------------------------------------------------------------------------- #
async def verify_step1_aoty_album(evaluator: Evaluator, parent_node, data: ResearchChainExtraction) -> None:
    """
    Step 1: Identify the album that won Album of the Year at the 2024 Grammys, with sources.
    """
    step_node = evaluator.add_parallel(
        id="step1_aoty_album",
        desc="Identify the album that won Album of the Year at the 2024 Grammy Awards.",
        parent=parent_node,
        critical=True,
    )

    album_name = data.album.name if (data.album and data.album.name) else ""
    album_sources = _safe_list(data.album.sources)

    # Leaf: Correctly identify the Album of the Year winning album
    aoty_identified = evaluator.add_leaf(
        id="aoty_album_identified",
        desc="Correctly identify the Album of the Year winning album at the 2024 Grammy Awards.",
        parent=step_node,
        critical=True,
    )
    claim = f"The album that won Album of the Year at the 2024 Grammy Awards is '{album_name}'."
    await evaluator.verify(
        claim=claim,
        node=aoty_identified,
        sources=album_sources,
        additional_instruction=(
            "Confirm from the provided URL(s) that the album explicitly won 'Album of the Year' "
            "at the 2024 Grammy Awards (the 66th Annual Grammy Awards)."
        ),
    )

    # Leaf: Provide valid reference URL(s) supporting the AOTY winning album (existence check)
    aoty_refs = evaluator.add_custom_node(
        result=len(album_sources) > 0,
        id="aoty_album_references",
        desc="Provide valid reference URL(s) supporting the 2024 Album of the Year winning album.",
        parent=step_node,
        critical=True,
    )


async def verify_step2_producer_award(evaluator: Evaluator, parent_node, data: ResearchChainExtraction) -> None:
    """
    Step 2: Identify the album’s producer (main/co-producer) who also won Producer of the Year (Non-Classical) at 2024 Grammys.
    """
    step_node = evaluator.add_parallel(
        id="step2_producer_with_award",
        desc="Identify the album’s main producer or co-producer who also won Producer of the Year (Non-Classical) at the same ceremony.",
        parent=parent_node,
        critical=True,
    )

    album_name = data.album.name if (data.album and data.album.name) else ""
    producer_name = data.producer.name if (data.producer and data.producer.name) else ""
    role_type = data.producer.role_type if (data.producer and data.producer.role_type) else "producer"
    role_sources = _safe_list(data.producer.role_sources if data.producer else [])
    award_sources = _safe_list(data.producer.award_sources if data.producer else [])

    # Leaf: Identify producer and confirm they are main/co-producer of the album
    producer_is_main_or_co = evaluator.add_leaf(
        id="producer_identified_and_is_main_or_coproducer",
        desc="Correctly identify the producer and confirm they are a main producer or co-producer of the identified album.",
        parent=step_node,
        critical=True,
    )
    claim_role = (
        f"{producer_name} is a {role_type} of the album '{album_name}', i.e., credited as a producer or co-producer."
    )
    await evaluator.verify(
        claim=claim_role,
        node=producer_is_main_or_co,
        sources=role_sources,
        additional_instruction=(
            "Verify from the provided URL(s) that this person is credited as a producer/co-producer on the specified album. "
            "Accept common synonyms like 'producer', 'co-producer', or 'produced by'."
        ),
    )

    # Leaf: Confirm producer won Producer of the Year (Non-Classical) at the 2024 Grammys
    producer_won_poty = evaluator.add_leaf(
        id="producer_won_poty_nonclassical_2024",
        desc="Confirm the identified producer won Producer of the Year (Non-Classical) at the 2024 Grammy Awards.",
        parent=step_node,
        critical=True,
    )
    claim_award = f"{producer_name} won 'Producer of the Year (Non-Classical)' at the 2024 Grammy Awards."
    await evaluator.verify(
        claim=claim_award,
        node=producer_won_poty,
        sources=award_sources,
        additional_instruction=(
            "Confirm from the provided URL(s) that the person explicitly won 'Producer of the Year (Non-Classical)' "
            "at the 2024 Grammys (66th Annual Grammy Awards)."
        ),
    )

    # Leaf: Provide valid reference URLs supporting both role and award (existence check)
    producer_refs = evaluator.add_custom_node(
        result=(len(role_sources) > 0 and len(award_sources) > 0),
        id="producer_references",
        desc="Provide valid reference URL(s) supporting the producer’s album role and Producer of the Year (Non-Classical) win.",
        parent=step_node,
        critical=True,
    )


async def verify_step3_producer_state(evaluator: Evaluator, parent_node, data: ResearchChainExtraction) -> None:
    """
    Step 3: Determine the U.S. state where the producer was born or raised, with supporting sources.
    """
    step_node = evaluator.add_parallel(
        id="step3_producer_state",
        desc="Determine the U.S. state where the producer was born or raised.",
        parent=parent_node,
        critical=True,
    )

    producer_name = data.producer.name if (data.producer and data.producer.name) else ""
    state_name = data.producer_state.state if (data.producer_state and data.producer_state.state) else ""
    state_sources = _safe_list(data.producer_state.sources if data.producer_state else [])

    # Leaf: Correctly identify the producer's birth/raised U.S. state
    producer_home_state = evaluator.add_leaf(
        id="producer_home_state",
        desc="Correctly identify the U.S. state where the producer was born or raised (as required by the question).",
        parent=step_node,
        critical=True,
    )
    claim_state = f"{producer_name} was born in or raised in the U.S. state of {state_name}."
    await evaluator.verify(
        claim=claim_state,
        node=producer_home_state,
        sources=state_sources,
        additional_instruction=(
            "Confirm from the provided URL(s) either birth state or raised state for the person. "
            "The task accepts either birth or raised state as correct."
        ),
    )

    # Leaf: Provide valid reference URLs for the state claim (existence check)
    state_refs = evaluator.add_custom_node(
        result=len(state_sources) > 0,
        id="state_references",
        desc="Provide valid reference URL(s) supporting the producer’s born-or-raised state.",
        parent=step_node,
        critical=True,
    )


async def verify_step4_venue_in_state(evaluator: Evaluator, parent_node, data: ResearchChainExtraction) -> None:
    """
    Step 4: Find a Guns N' Roses 2026 North American tour venue located in the producer’s state, with sources.
    """
    step_node = evaluator.add_parallel(
        id="step4_tour_venue_in_state",
        desc="Find the Guns N' Roses 2026 North American tour venue located in the producer’s state.",
        parent=parent_node,
        critical=True,
    )

    venue_name = data.venue.name if (data.venue and data.venue.name) else ""
    producer_state_name = data.producer_state.state if (data.producer_state and data.producer_state.state) else ""
    tour_sources = _safe_list(data.venue.tour_sources if data.venue else [])

    # Leaf: Correctly identify a tour venue located in the identified state
    venue_is_on_tour_in_state = evaluator.add_leaf(
        id="venue_on_2026_tour_in_state",
        desc="Correctly identify a Guns N' Roses 2026 North American tour venue that is located in the identified state.",
        parent=step_node,
        critical=True,
    )
    claim_venue_tour = (
        f"The Guns N' Roses 2026 North American tour includes a concert at '{venue_name}', and this venue is located in {producer_state_name}."
    )
    await evaluator.verify(
        claim=claim_venue_tour,
        node=venue_is_on_tour_in_state,
        sources=tour_sources,
        additional_instruction=(
            "Confirm from the provided URL(s) that the venue is part of Guns N' Roses 2026 North American tour "
            "and that the venue is located in the specified state."
        ),
    )

    # Leaf: Provide valid reference URL(s) supporting that the venue is on the tour and is in the identified state (existence check)
    venue_tour_refs = evaluator.add_custom_node(
        result=len(tour_sources) > 0,
        id="venue_tour_references",
        desc="Provide valid reference URL(s) supporting that the identified venue is on the Guns N' Roses 2026 North American tour and is located in the identified state.",
        parent=step_node,
        critical=True,
    )


async def verify_step5_venue_details(evaluator: Evaluator, parent_node, data: ResearchChainExtraction) -> None:
    """
    Step 5: Provide venue details: concert date, concert seating capacity, and venue city/state, each with sources.
    """
    step_node = evaluator.add_parallel(
        id="step5_venue_details",
        desc="Provide the required details for the identified venue on the tour: concert date, concert seating capacity, and venue city/state.",
        parent=parent_node,
        critical=True,
    )

    venue_name = data.venue.name if (data.venue and data.venue.name) else ""
    concert_date = data.venue_details.concert_date if (data.venue_details and data.venue_details.concert_date) else ""
    date_sources = _safe_list(data.venue_details.date_sources if data.venue_details else [])

    capacity = data.venue_details.capacity if (data.venue_details and data.venue_details.capacity) else ""
    capacity_sources = _safe_list(data.venue_details.capacity_sources if data.venue_details else [])

    city = data.venue_details.city if (data.venue_details and data.venue_details.city) else ""
    state = data.venue_details.state if (data.venue_details and data.venue_details.state) else ""
    location_sources = _safe_list(data.venue_details.location_sources if data.venue_details else [])

    # Concert date exact
    concert_date_exact = evaluator.add_leaf(
        id="concert_date_exact",
        desc="Provide the exact concert date at the venue (month, day, and year).",
        parent=step_node,
        critical=True,
    )
    claim_date = f"The concert date for Guns N' Roses at '{venue_name}' on the 2026 North American tour is {concert_date}."
    await evaluator.verify(
        claim=claim_date,
        node=concert_date_exact,
        sources=date_sources,
        additional_instruction=(
            "Confirm from the provided URL(s) that the concert date includes month, day, and year, and matches exactly."
        ),
    )

    concert_date_refs = evaluator.add_custom_node(
        result=len(date_sources) > 0,
        id="concert_date_references",
        desc="Provide valid reference URL(s) supporting the concert date.",
        parent=step_node,
        critical=True,
    )

    # Concert capacity
    concert_capacity = evaluator.add_leaf(
        id="concert_capacity",
        desc="Provide the venue seating capacity specifically for concerts.",
        parent=step_node,
        critical=True,
    )
    claim_capacity = f"The concert seating capacity at '{venue_name}' is {capacity}."
    await evaluator.verify(
        claim=claim_capacity,
        node=concert_capacity,
        sources=capacity_sources,
        additional_instruction=(
            "From the provided URL(s), confirm the seating capacity specifically for concerts (not necessarily for sports). "
            "If multiple capacities are listed, use the concert configuration."
        ),
    )

    capacity_refs = evaluator.add_custom_node(
        result=len(capacity_sources) > 0,
        id="capacity_references",
        desc="Provide valid reference URL(s) supporting the venue’s concert seating capacity.",
        parent=step_node,
        critical=True,
    )

    # Venue city and state
    venue_city_state = evaluator.add_leaf(
        id="venue_city_and_state",
        desc="Provide the venue’s city and state location.",
        parent=step_node,
        critical=True,
    )
    claim_location = f"The venue '{venue_name}' is located in {city}, {state}."
    await evaluator.verify(
        claim=claim_location,
        node=venue_city_state,
        sources=location_sources,
        additional_instruction=(
            "Confirm from the provided URL(s) the city and state for the venue."
        ),
    )

    location_refs = evaluator.add_custom_node(
        result=len(location_sources) > 0,
        id="location_references",
        desc="Provide valid reference URL(s) supporting the venue’s city and state location.",
        parent=step_node,
        critical=True,
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
    Evaluate an answer for the Grammys-to-Guns N' Roses chain task using the Mind2Web2 framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Framework root remains non-critical; we'll create a critical sequential chain under it
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

    # Create a critical sequential chain root to enforce step ordering and gating
    chain_root = evaluator.add_sequential(
        id="chain_root",
        desc="Complete the multi-step research chain from 2024 Grammys Album of the Year to a Guns N' Roses 2026 tour venue and required venue details, with supporting reference URLs.",
        parent=root,
        critical=True,
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_chain(),
        template_class=ResearchChainExtraction,
        extraction_name="research_chain_extraction",
    )

    # Add a quick custom info snapshot for debugging
    evaluator.add_custom_info(
        info={
            "album": extracted.album.dict() if extracted.album else None,
            "producer": extracted.producer.dict() if extracted.producer else None,
            "producer_state": extracted.producer_state.dict() if extracted.producer_state else None,
            "venue": extracted.venue.dict() if extracted.venue else None,
            "venue_details": extracted.venue_details.dict() if extracted.venue_details else None,
        },
        info_type="extraction_snapshot",
        info_name="extraction_overview",
    )

    # Build and verify each step under the critical sequential chain
    await verify_step1_aoty_album(evaluator, chain_root, extracted)
    await verify_step2_producer_award(evaluator, chain_root, extracted)
    await verify_step3_producer_state(evaluator, chain_root, extracted)
    await verify_step4_venue_in_state(evaluator, chain_root, extracted)
    await verify_step5_venue_details(evaluator, chain_root, extracted)

    # Return structured summary
    return evaluator.get_summary()