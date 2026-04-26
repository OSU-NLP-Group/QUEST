import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_outdoor_amphitheaters_waco_ca"
TASK_DESCRIPTION = """
I am researching outdoor concert venues for a music festival planning project. I need to identify three outdoor amphitheaters in the United States that meet the following criteria: (1) Each venue must be located in Washington, Colorado, or California; (2) Each venue must have a total capacity between 15,000 and 30,000 people; (3) Each venue must be classified as an outdoor amphitheater (not an indoor arena); (4) Each venue must have both permanent/reserved seating and lawn seating areas; (5) For each venue, provide information about its management or operating company; (6) For each venue, provide a reference URL from the venue's official website or a reliable source that confirms the venue information. Please provide the name, location, capacity, seating configuration, management information, and reference URL for each of the three venues.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow ranges/approx (e.g., "approx. 18,000")
    classification: Optional[str] = None  # e.g., "outdoor amphitheater"
    seating_configuration: Optional[str] = None  # free text summary from the answer
    has_reserved_or_permanent_seating: Optional[bool] = None
    has_lawn_seating: Optional[bool] = None
    management_or_operator: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to three (3) outdoor amphitheater venues from the answer. Each venue should be a distinct entry.
    For each venue, extract the following fields:
    - name: The venue name as written in the answer.
    - city: The city where the venue is located, if stated.
    - state: The U.S. state abbreviation or full name where the venue is located, if stated (e.g., "WA", "Washington").
    - capacity: The total capacity as stated in the answer (string; keep formatting such as "about 18,000", "27,500", "15k-20k").
    - classification: The venue classification as stated in the answer (e.g., "outdoor amphitheater", "open-air amphitheatre").
    - seating_configuration: Any description of seating types, e.g., "reserved seating + lawn", "permanent seats and lawn".
    - has_reserved_or_permanent_seating: true/false, whether the answer indicates there are permanent or reserved seats.
    - has_lawn_seating: true/false, whether the answer indicates there is a lawn or general admission lawn area.
    - management_or_operator: The management or operating company named in the answer (e.g., "Live Nation", "AEG Presents", "City of ..."), if provided.
    - reference_urls: An array of all URLs cited in the answer for this venue. Extract only valid URLs that appear in the answer. If none are given, return an empty array.
    
    Notes:
    - Do not invent values not explicitly in the answer text.
    - If a field is missing in the answer, set it to null (or empty array for reference_urls).
    - Return a JSON object: { "venues": [ ... up to 3 items ... ] }.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n + 1}"


def _safe_name(venue: VenueItem, idx: int) -> str:
    return venue.name or f"Venue #{idx + 1}"


# --------------------------------------------------------------------------- #
# Verification for one venue                                                  #
# --------------------------------------------------------------------------- #
async def verify_one_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int,
) -> None:
    """
    Build verification subtree for a single venue according to the rubric.
    """
    group_node = evaluator.add_parallel(
        id=f"venue_{idx + 1}",
        desc=f"{ordinal(idx)} outdoor amphitheater meeting all criteria",
        parent=parent_node,
        critical=False,  # Venue-level partial credit is allowed
    )

    # Prepare convenience values
    name = _safe_name(venue, idx)
    loc_fragment = ", ".join([p for p in [venue.city, venue.state] if p]) or (venue.state or "")

    # 1) Reference URL validity/reliability (critical)
    #    If no URLs are provided, immediately fail this item.
    ref_desc = "A valid reference URL is provided from the venue's official website or a reliable source"
    if not venue.reference_urls:
        evaluator.add_custom_node(
            result=False,
            id=f"venue_{idx + 1}_reference",
            desc=ref_desc,
            parent=group_node,
            critical=True,
        )
        # No further checks will be skipped automatically only after a failure is registered.
        # Continue to create remaining nodes; the framework will skip subsequent critical siblings after a failure.
    else:
        ref_node = evaluator.add_leaf(
            id=f"venue_{idx + 1}_reference",
            desc=ref_desc,
            parent=group_node,
            critical=True,
        )
        ref_claim = (
            f"At least one of the provided URLs is the official website of {name} or another highly reliable source "
            f"(e.g., a recognized operator like Live Nation/AEG, or an authoritative city/government/educational page) "
            f"and it contains venue information about {name}."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_node,
            sources=venue.reference_urls,
            additional_instruction=(
                "Assess reliability primarily by domain ownership and explicit venue information on the page. "
                "Official venue domains, major operators (Live Nation/AEG), prominent ticketing platforms (Ticketmaster), "
                "and authoritative civic/education domains are acceptable. Blog posts or generic tourism aggregators are weaker."
            ),
        )

    # 2) Management/operating company provided in the answer (critical, existence check)
    evaluator.add_custom_node(
        result=bool(venue.management_or_operator and str(venue.management_or_operator).strip()),
        id=f"venue_{idx + 1}_management",
        desc="Information about the venue's management or operating company is provided",
        parent=group_node,
        critical=True,
    )

    # 3) Capacity between 15,000 and 30,000 (critical)
    cap_node = evaluator.add_leaf(
        id=f"venue_{idx + 1}_capacity",
        desc="The venue has a total capacity between 15,000 and 30,000 people",
        parent=group_node,
        critical=True,
    )
    cap_claim = (
        f"The total capacity of {name} is between 15,000 and 30,000 people (inclusive), considering total capacity "
        f"including both fixed/reserved seats and lawn/general admission areas as applicable."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=venue.reference_urls,
        additional_instruction="Use the cited page(s) to confirm overall/total capacity. Approximations such as 'about 18,000' are acceptable if clearly within range.",
    )

    # 4) State is WA, CO, or CA (critical)
    st_node = evaluator.add_leaf(
        id=f"venue_{idx + 1}_state",
        desc="The venue is located in Washington, Colorado, or California",
        parent=group_node,
        critical=True,
    )
    st_claim = (
        f"{name} is located in one of these U.S. states: Washington, Colorado, or California. "
        f"The answer lists its location as {loc_fragment}."
    )
    await evaluator.verify(
        claim=st_claim,
        node=st_node,
        sources=venue.reference_urls,
        additional_instruction="Confirm the venue's state from the page. Minor city naming variations are fine; focus on the state membership (WA/CO/CA).",
    )

    # 5) Outdoor amphitheater classification (critical)
    out_node = evaluator.add_leaf(
        id=f"venue_{idx + 1}_outdoor",
        desc="The venue is classified as an outdoor amphitheater",
        parent=group_node,
        critical=True,
    )
    out_claim = (
        f"{name} is an outdoor amphitheater (i.e., open-air), not an indoor arena."
    )
    await evaluator.verify(
        claim=out_claim,
        node=out_node,
        sources=venue.reference_urls,
        additional_instruction="Look for terms like 'outdoor amphitheater', 'open-air', 'amphitheatre', or explicit indications it is not enclosed/indoor.",
    )

    # 6) Seating configuration includes both permanent/reserved seating and a lawn (critical)
    seat_node = evaluator.add_leaf(
        id=f"venue_{idx + 1}_seating_types",
        desc="The venue has both permanent/reserved seating and lawn seating areas",
        parent=group_node,
        critical=True,
    )
    seat_claim = (
        f"{name} offers both (a) permanent/reserved/fixed seating and (b) lawn or general admission lawn seating."
    )
    await evaluator.verify(
        claim=seat_claim,
        node=seat_node,
        sources=venue.reference_urls,
        additional_instruction="Check seating descriptions, seat maps, or venue overview sections for mentions of fixed/reserved seats AND lawn/GA lawn.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the outdoor amphitheaters task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root-level venues evaluated independently
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

    # Extract venue candidates from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Keep exactly three venues (pad with empty entries if fewer)
    venues: List[VenueItem] = list(extracted.venues[:3]) if extracted and extracted.venues else []
    while len(venues) < 3:
        venues.append(VenueItem())

    # Build verification tree for each of the three target venues
    for i in range(3):
        await verify_one_venue(evaluator, root, venues[i], i)

    # Return summary with verification tree and scores
    return evaluator.get_summary()