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
TASK_ID = "jim_gaffigan_2026_east_large_venues"
TASK_DESCRIPTION = """
A group planning to attend comedian Jim Gaffigan's 'Everything is Wonderful!' 2026 tour is seeking venues that can accommodate large audiences and provide accessibility features. From his tour schedule, identify three venues that meet ALL of the following requirements: (1) Located in the Eastern United States (states east of the Mississippi River), (2) Minimum seating capacity of 6,000, (3) Jim Gaffigan performances scheduled between February 1 and March 31, 2026, (4) Provide ADA-compliant wheelchair accessible seating with companion seats available. For each venue, provide: the venue name, city and state, seating capacity, performance date(s), and a reference URL confirming the information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to maximize compatibility
    performance_dates: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to three venues from the answer that correspond to Jim Gaffigan's 'Everything is Wonderful!' 2026 tour.
    For each venue, extract the following fields exactly as stated in the answer:
    - venue_name: The venue's official name (e.g., 'Madison Square Garden').
    - city: The city where the venue is located.
    - state: The U.S. state where the venue is located (use the state's full name or common abbreviation exactly as provided).
    - capacity: The stated seating capacity value for the venue (keep it as a string as written in the answer; do not convert to a number).
    - performance_dates: List of date strings (e.g., 'March 2, 2026' or '03/02/2026') for Jim Gaffigan's performance(s) at this venue as provided in the answer.
    - reference_urls: All URLs cited that support or reference the venue, event, capacity, accessibility, or tour info. Include official venue pages, event listing pages, ticketing pages, or Jim Gaffigan's official tour page URLs. Extract only valid URLs mentioned in the answer text.
    
    Rules:
    - Return a JSON object with a 'venues' array containing up to 3 venue objects ordered as they appear in the answer.
    - If a field is missing for a venue, set it to null (for strings) or an empty array (for lists).
    - Do not invent information; only extract what's explicitly in the answer.
    - For URLs, include the full URL. If a URL is missing a protocol, prepend 'http://'.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0 and any(isinstance(u, str) and u.strip() for u in urls))


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueInfo,
    ordinal_index: int,
) -> None:
    """
    Build verification subtree for a single venue (Venue_1 / Venue_2 / Venue_3).
    All existence checks are marked critical to gate subsequent verifications.
    Evidence-grounded checks use the provided reference URLs.
    """
    v_id = f"Venue_{ordinal_index}"
    v_node = evaluator.add_parallel(
        id=v_id,
        desc=f"{['First','Second','Third'][ordinal_index-1]} qualifying venue meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Existence checks (critical)
    evaluator.add_custom_node(
        result=_has_text(venue.venue_name),
        id=f"{v_id}_Name",
        desc="A specific venue name is provided",
        parent=v_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(venue.city),
        id=f"{v_id}_City",
        desc="The city where the venue is located is provided",
        parent=v_node,
        critical=True
    )

    # Separate existence for state (to avoid aggregating multiple checks in one leaf)
    state_exists_node = evaluator.add_custom_node(
        result=_has_text(venue.state),
        id=f"{v_id}_State_Provided",
        desc="The state where the venue is located is provided",
        parent=v_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_text(venue.capacity),
        id=f"{v_id}_Capac ity_Value".replace(" ", ""),  # ensure id without spaces
        desc="The actual seating capacity of the venue is provided",
        parent=v_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(venue.performance_dates) > 0,
        id=f"{v_id}_Date_Specific",
        desc="Specific performance date(s) for Jim Gaffigan at this venue are provided",
        parent=v_node,
        critical=True
    )

    ref_exists_node = evaluator.add_custom_node(
        result=_has_urls(venue.reference_urls),
        id=f"{v_id}_Reference",
        desc="A reference URL supporting the venue information is provided",
        parent=v_node,
        critical=True
    )

    # 2) Eastern United States check (simple verify using general knowledge)
    state_check_leaf = evaluator.add_leaf(
        id=f"{v_id}_State",
        desc="The state where the venue is located is in the Eastern United States (east of the Mississippi River)",
        parent=v_node,
        critical=True
    )
    state_claim = f"The U.S. state '{venue.state or ''}' is east of the Mississippi River."
    await evaluator.verify(
        claim=state_claim,
        node=state_check_leaf,
        additional_instruction="Use general knowledge of U.S. geography to determine if the named state lies east of the Mississippi River. Consider states whose territory lies entirely or primarily east of the river as 'east'. Include DC as east.",
        extra_prerequisites=[state_exists_node]
    )

    # 3) Capacity minimum check – evidence grounded
    capacity_min_leaf = evaluator.add_leaf(
        id=f"{v_id}_Capacity_Minimum",
        desc="The venue has a seating capacity of at least 6,000",
        parent=v_node,
        critical=True
    )
    capacity_min_claim = "This venue's seating capacity is at least 6,000."
    await evaluator.verify(
        claim=capacity_min_claim,
        node=capacity_min_leaf,
        sources=venue.reference_urls,
        additional_instruction="From the venue's official page or other reliable sources, confirm that the stated seating capacity (fixed or typical event configuration) is ≥ 6,000. If capacity is given as a range or 'up to', treat 'at least 6,000' as satisfied only if the minimum or typical capacity is ≥ 6,000."
    )

    # 4) Date window check – evidence grounded (Feb 1 to Mar 31, 2026 inclusive)
    date_range_leaf = evaluator.add_leaf(
        id=f"{v_id}_Date_Range",
        desc="Jim Gaffigan has performances at this venue scheduled between February 1 and March 31, 2026",
        parent=v_node,
        critical=True
    )
    date_range_claim = "At this venue, Jim Gaffigan's performance date(s) fall(s) between February 1 and March 31, 2026 (inclusive)."
    await evaluator.verify(
        claim=date_range_claim,
        node=date_range_leaf,
        sources=venue.reference_urls,
        additional_instruction="Check the event listing(s) for Jim Gaffigan at this venue and confirm that at least one scheduled performance date is within Feb 1–Mar 31, 2026 inclusive. If multiple dates exist, it's sufficient that at least one date falls in that window."
    )

    # 5) Tour confirmation – evidence grounded
    tour_leaf = evaluator.add_leaf(
        id=f"{v_id}_Tour",
        desc="The venue is confirmed to be part of Jim Gaffigan's 'Everything is Wonderful!' 2026 tour",
        parent=v_node,
        critical=True
    )
    tour_claim = "This venue's Jim Gaffigan performance is part of the 'Everything is Wonderful!' 2026 tour."
    await evaluator.verify(
        claim=tour_claim,
        node=tour_leaf,
        sources=venue.reference_urls,
        additional_instruction="Look for explicit mention of 'Everything is Wonderful!' and/or '2026 tour' on the official tour page, venue page, or ticket listing to confirm association."
    )

    # 6) Accessibility checks – evidence grounded
    wheelchair_leaf = evaluator.add_leaf(
        id=f"{v_id}_Wheelchair",
        desc="The venue provides ADA-compliant wheelchair accessible seating",
        parent=v_node,
        critical=True
    )
    wheelchair_claim = "This venue provides ADA-compliant wheelchair accessible seating."
    await evaluator.verify(
        claim=wheelchair_claim,
        node=wheelchair_leaf,
        sources=venue.reference_urls,
        additional_instruction="Check the venue's accessibility or ticketing policies to confirm wheelchair-accessible seating exists and is ADA-compliant."
    )

    companion_leaf = evaluator.add_leaf(
        id=f"{v_id}_Companion",
        desc="The venue offers companion seats (up to 3 additional seats) adjacent to or near wheelchair accessible seating",
        parent=v_node,
        critical=True
    )
    companion_claim = "This venue offers companion seats (up to 3 additional seats) adjacent to or near wheelchair accessible seating."
    await evaluator.verify(
        claim=companion_claim,
        node=companion_leaf,
        sources=venue.reference_urls,
        additional_instruction="Look for language indicating companion seating availability adjacent to wheelchair areas; typical policies allow 1–3 companion seats. Equivalent phrasing like 'companion tickets' or 'adjacent companion seating' is acceptable."
    )

    price_levels_leaf = evaluator.add_leaf(
        id=f"{v_id}_Price_Levels",
        desc="Accessible seating is available at multiple price levels",
        parent=v_node,
        critical=True
    )
    price_levels_claim = "Accessible seating is available at multiple price levels at this venue."
    await evaluator.verify(
        claim=price_levels_claim,
        node=price_levels_leaf,
        sources=venue.reference_urls,
        additional_instruction="Verify the venue states that accessible seating is offered across multiple price categories/levels or sections; acceptable phrasing includes 'available in various price levels', 'across multiple sections', or equivalent."
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
    Evaluate the agent's answer for the Jim Gaffigan 'Everything is Wonderful!' 2026 tour venue selection task.
    """
    # Initialize evaluator with parallel aggregation at the root
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

    # Extract venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="extracted_venues"
    )

    # Prepare up to 3 venues (pad with empty if fewer)
    venues: List[VenueInfo] = list(extracted.venues[:3])
    while len(venues) < 3:
        venues.append(VenueInfo())

    # Build verification tree under root
    qualifying_root = evaluator.add_parallel(
        id="Qualifying_Tour_Venues",
        desc="Evaluate whether the provided venues from Jim Gaffigan's 2026 tour meet all specified criteria",
        parent=root,
        critical=False
    )

    # Verify each venue in parallel subtree
    for idx in range(3):
        await verify_single_venue(
            evaluator=evaluator,
            parent_node=qualifying_root,
            venue=venues[idx],
            ordinal_index=idx + 1
        )

    # Return structured evaluation summary
    return evaluator.get_summary()