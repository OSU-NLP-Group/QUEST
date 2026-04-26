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
TASK_ID = "tour_venues_2026"
TASK_DESCRIPTION = (
    "A professional music artist is planning a 2026 concert tour across the United States and needs to identify "
    "suitable major indoor concert venues in four different metropolitan areas. For each of the following cities - "
    "New York City, Chicago, Los Angeles, and Boston - identify one major indoor arena that meets all of these "
    "requirements: (1) Has a concert seating capacity between 15,000 and 25,000 people, (2) Is classified as an arena "
    "(not a theater, club, or amphitheater), (3) Is an indoor venue (not an outdoor or open-air facility), "
    "(4) Is located within the city limits or immediate metropolitan area of the specified city. For each identified "
    "venue, provide: (a) the venue name, (b) its concert capacity, (c) confirmation that it is an indoor arena, and "
    "(d) a reference URL from a reliable source (such as the venue's official website, Wikipedia, or a reputable "
    "news/entertainment source) that confirms these specifications."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueEntry(BaseModel):
    """Information for a single city's selected venue."""
    venue_name: Optional[str] = None
    concert_capacity: Optional[str] = None  # keep as string to handle ranges or textual descriptions
    is_indoor_arena_confirmation: Optional[str] = None  # e.g., "indoor arena", "indoor"
    venue_type: Optional[str] = None  # e.g., "arena", "stadium", etc.
    location_city_or_area: Optional[str] = None  # e.g., "Manhattan, New York City" or "Inglewood, CA"
    reference_urls: List[str] = Field(default_factory=list)  # extract actual URLs mentioned in the answer


class TourVenuesExtraction(BaseModel):
    """Structured extraction covering the four required cities."""
    new_york_city: Optional[VenueEntry] = None
    chicago: Optional[VenueEntry] = None
    los_angeles: Optional[VenueEntry] = None
    boston: Optional[VenueEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_tour_venues() -> str:
    return """
    From the answer, extract one suitable major indoor concert arena for each of the four cities:
    New York City, Chicago, Los Angeles, and Boston.

    For each city, extract the following fields if available:
    - venue_name: The full official venue name mentioned for the city.
    - concert_capacity: The concert seating capacity mentioned (as text, keep ranges or approximations exactly).
    - is_indoor_arena_confirmation: Any confirmation text indicating the venue is indoor and an arena (e.g., "indoor arena").
    - venue_type: The venue classification provided (e.g., "arena", "stadium", "theater").
    - location_city_or_area: The location text given in the answer (e.g., neighborhood, city, or immediate metro area).
    - reference_urls: All URLs provided for the venue (official site, Wikipedia, or reputable media). Extract actual URLs
      explicitly present in the answer; include full URLs. If a URL is missing protocol, prepend http://.

    IMPORTANT:
    - If multiple venues are listed for a city, choose the first one that best matches the requested criteria.
    - If some fields are missing for a city, set them to null, and return an empty list for reference_urls if none are present.
    - Only include URLs explicitly mentioned in the answer (plain URLs or markdown links).
    - Return a JSON object with keys: new_york_city, chicago, los_angeles, boston. Each key maps to an object with the
      specified fields. If a city's venue is not provided, set that city to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate & strip
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Verification logic per city                                                 #
# --------------------------------------------------------------------------- #
async def verify_city(
    evaluator: Evaluator,
    parent_node,
    city_key: str,
    city_label: str,
    city_node_id: str,
    extracted: TourVenuesExtraction,
) -> None:
    """
    Build verification sub-tree and run checks for one city.

    Parameters
    ----------
    evaluator : Evaluator
        Evaluation executor.
    parent_node : VerificationNode
        Parent node to attach this city's verification.
    city_key : str
        Attribute key in TourVenuesExtraction (e.g., 'new_york_city').
    city_label : str
        Human city label (e.g., 'New York City').
    city_node_id : str
        Node ID to use for the city parent (e.g., 'New_York_City_Venue').
    extracted : TourVenuesExtraction
        Extracted data from the answer.
    """
    city_entry: Optional[VenueEntry] = getattr(extracted, city_key)
    venue_name = (city_entry.venue_name or "").strip() if city_entry else ""
    urls = _safe_list(city_entry.reference_urls if city_entry else [])

    # Create city parent node (parallel aggregation, non-critical)
    city_node = evaluator.add_parallel(
        id=city_node_id,
        desc=f"Identification of a suitable arena in {city_label} metropolitan area",
        parent=parent_node,
        critical=False
    )

    # Existence & basic prerequisites (custom critical check to gate downstream verifications)
    # We require a venue name and at least one reference URL to proceed meaningfully.
    evaluator.add_custom_node(
        result=(bool(venue_name) and len(urls) > 0),
        id=f"{city_node_id}_required_info",
        desc=f"{city_label}: Venue name and at least one reference URL are provided",
        parent=city_node,
        critical=True
    )

    # 1) Capacity requirement (critical)
    cap_node = evaluator.add_leaf(
        id=f"{city_node_id.split('_')[0]}_Capacity_Requirement" if "Capacity_Requirement" in city_node_id else f"{city_node_id}_Capacity_Requirement",
        desc=f"The venue has a concert capacity between 15,000 and 25,000",
        parent=city_node,
        critical=True
    )
    cap_claim = (
        f"The venue '{venue_name}' has a concert seating capacity between 15,000 and 25,000 people."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=urls,
        additional_instruction=(
            "Use the provided source(s) to check the venue's concert capacity. "
            "It's acceptable if the page lists a general or maximum capacity but clearly indicates concert/event capacity "
            "within 15,000–25,000. Prefer concert capacity; if only overall/event capacity is available and lies in this range, accept."
        ),
    )

    # 2) Indoor classification (critical)
    indoor_node = evaluator.add_leaf(
        id=f"{city_node_id.split('_')[0]}_Indoor_Classification" if "Indoor_Classification" in city_node_id else f"{city_node_id}_Indoor_Classification",
        desc="The venue is an indoor facility",
        parent=city_node,
        critical=True
    )
    indoor_claim = f"The venue '{venue_name}' is an indoor facility (not outdoor or open-air)."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=urls,
        additional_instruction=(
            "Confirm from the page content that the venue is indoors. For modern arenas, the structure and descriptions typically imply indoor use; "
            "do not accept outdoor-only amphitheaters or open-air stadiums."
        ),
    )

    # 3) Arena type classification (critical)
    arena_type_node = evaluator.add_leaf(
        id=f"{city_node_id.split('_')[0]}_Arena_Type" if "Arena_Type" in city_node_id else f"{city_node_id}_Arena_Type",
        desc="The venue is classified as an arena, not a theater, club, or amphitheater",
        parent=city_node,
        critical=True
    )
    arena_type_claim = (
        f"The venue '{venue_name}' is classified as an arena (and not as a theater, club, or amphitheater)."
    )
    await evaluator.verify(
        claim=arena_type_claim,
        node=arena_type_node,
        sources=urls,
        additional_instruction=(
            "Verify the venue's classification on the page (e.g., explicitly 'arena'). "
            "Do not accept theaters, clubs, amphitheaters, or outdoor stadiums."
        ),
    )

    # 4) Location verification (critical)
    loc_node = evaluator.add_leaf(
        id=f"{city_node_id.split('_')[0]}_Location_Verification" if "Location_Verification" in city_node_id else f"{city_node_id}_Location_Verification",
        desc=f"The venue is located within {city_label} or its immediate metropolitan area",
        parent=city_node,
        critical=True
    )
    # Allow common immediate metro examples (e.g., Elmont for NYC, Rosemont for Chicago, Inglewood for LA)
    loc_claim = (
        f"The venue '{venue_name}' is located within {city_label} city limits or the immediate metropolitan area."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=urls,
        additional_instruction=(
            "Confirm the venue's location on the page. Immediate metro areas include adjacent municipalities commonly considered part of the metro "
            "(e.g., Elmont for NYC, Rosemont for Chicago, Inglewood for Los Angeles). "
            "If the page indicates the venue is in such adjacent areas recognized within the metro, accept."
        ),
    )

    # 5) Reliable reference URL provided (critical)
    ref_node = evaluator.add_leaf(
        id=f"{city_node_id.split('_')[0]}_Reference_URL" if "Reference_URL" in city_node_id else f"{city_node_id}_Reference_URL",
        desc="A reliable reference URL is provided confirming the venue specifications",
        parent=city_node,
        critical=True
    )
    ref_claim = (
        "At least one of the provided URLs is a reliable source (official venue site, Wikipedia, or reputable news/entertainment publication) "
        "and is suitable for confirming venue specifications."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=urls,
        additional_instruction=(
            "Assess reliability primarily by the domain and page context: official venue websites, Wikipedia.org, and recognized "
            "news/entertainment publications are reliable. Aggregator sites or random blogs are typically not. "
            "You only need to determine that at least one provided URL is reliable and appropriate to confirm the venue's details."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2026 tour venues identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Cities are independent; allow partial credit across them
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

    # Extract structured venues info
    extracted = await evaluator.extract(
        prompt=prompt_extract_tour_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="tour_venues_extraction",
    )

    # Add a top-level grouping node for clarity (non-critical, parallel)
    tour_node = evaluator.add_parallel(
        id="Tour_Venues_Identification",
        desc="Identification of suitable major indoor concert arenas in four US cities for a music tour",
        parent=root,
        critical=False
    )

    # Optional ground-truth-style info (criteria)
    evaluator.add_ground_truth({
        "cities": ["New York City", "Chicago", "Los Angeles", "Boston"],
        "requirements": {
            "capacity": "Concert seating capacity between 15,000 and 25,000",
            "type": "Arena (not theater, club, amphitheater)",
            "indoor": "Indoor facility",
            "location": "Within city limits or immediate metropolitan area",
            "source": "Reliable reference URL (official site, Wikipedia, reputable news/entertainment)"
        }
    }, gt_type="criteria")

    # Verify each city
    await verify_city(
        evaluator=evaluator,
        parent_node=tour_node,
        city_key="new_york_city",
        city_label="New York City",
        city_node_id="New_York_City_Venue",
        extracted=extracted,
    )
    await verify_city(
        evaluator=evaluator,
        parent_node=tour_node,
        city_key="chicago",
        city_label="Chicago",
        city_node_id="Chicago_Venue",
        extracted=extracted,
    )
    await verify_city(
        evaluator=evaluator,
        parent_node=tour_node,
        city_key="los_angeles",
        city_label="Los Angeles",
        city_node_id="Los_Angeles_Venue",
        extracted=extracted,
    )
    await verify_city(
        evaluator=evaluator,
        parent_node=tour_node,
        city_key="boston",
        city_label="Boston",
        city_node_id="Boston_Venue",
        extracted=extracted,
    )

    # Return structured summary
    return evaluator.get_summary()