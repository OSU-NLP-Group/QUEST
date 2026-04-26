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
TASK_ID = "lv_residency_venue"
TASK_DESCRIPTION = (
    "A music artist is planning a residency in Las Vegas and requires a venue with a seating capacity between 4,000 and 6,000 seats. "
    "Identify one such venue in Las Vegas that meets this capacity requirement. Provide the venue's official name, exact seating capacity, and complete street address."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    official_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    street_address: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract venue information exactly as presented in the answer text. The task asks for one Las Vegas venue with 4,000–6,000 seating capacity, but the answer may mention multiple venues. Extract all venues mentioned and we will select the first one later.

    For each venue mentioned, extract the following fields:
    - official_name: The official name of the venue as written in the answer.
    - seating_capacity: The specific seating capacity number as stated in the answer (keep the exact string, e.g., "4,100" or "4100"). If only a range is provided without a single number, still return the string exactly as written.
    - street_address: The complete street address as written in the answer (e.g., "3570 S Las Vegas Blvd, Las Vegas, NV 89109").
    - source_urls: A list of all URLs cited in the answer that support information about this venue (official site, Wikipedia, venue page, ticketing pages, etc.). Only include URLs explicitly present in the answer.

    Return a JSON object with:
    {
      "venues": [
        {
          "official_name": "...",
          "seating_capacity": "...",
          "street_address": "...",
          "source_urls": ["...", "..."]
        },
        ...
      ]
    }

    If any field is missing for a venue, set it to null (or an empty list for source_urls).
    Do not invent or infer any information not explicitly stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def select_primary_venue(extraction: VenuesExtraction) -> VenueItem:
    """Select the first venue mentioned or return an empty placeholder if none."""
    if extraction and extraction.venues:
        return extraction.venues[0]
    return VenueItem()


def looks_like_address(address: Optional[str]) -> bool:
    """Lightweight heuristic: presence of a digit and a comma suggests a street address."""
    if not address:
        return False
    has_digit = any(ch.isdigit() for ch in address)
    has_comma = "," in address
    return has_digit and has_comma


def has_numeric_token(text: Optional[str]) -> bool:
    """Check if the string contains at least one digit (for capacity presence)."""
    if not text:
        return False
    return any(ch.isdigit() for ch in text)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree_for_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem
) -> None:
    """
    Create the verification tree nodes for the venue and run verifications.
    Structure mirrors the rubric: a single parallel node with six critical leaf checks.
    """
    # Parallel aggregation node for the complete venue identification
    complete_node = evaluator.add_parallel(
        id="Complete_Venue_Identification",
        desc="Correctly identify a Las Vegas entertainment venue meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Presence checks (custom, critical)
    name_provided_node = evaluator.add_custom_node(
        result=(venue.official_name is not None and venue.official_name.strip() != ""),
        id="Venue_Name_Provided",
        desc="The official name of the venue is clearly stated",
        parent=complete_node,
        critical=True
    )

    capacity_stated_node = evaluator.add_custom_node(
        result=has_numeric_token(venue.seating_capacity),
        id="Exact_Capacity_Stated",
        desc="The specific seating capacity number is provided",
        parent=complete_node,
        critical=True
    )

    address_provided_node = evaluator.add_custom_node(
        result=(venue.street_address is not None and venue.street_address.strip() != ""),
        id="Street_Address_Provided",
        desc="The complete street address of the venue is included",
        parent=complete_node,
        critical=True
    )

    # Evidence-based checks (critical)
    # 1) Geographic_Location
    location_node = evaluator.add_leaf(
        id="Geographic_Location",
        desc="The venue is located in Las Vegas, Nevada",
        parent=complete_node,
        critical=True
    )
    location_claim = f"The venue '{venue.official_name}' is located in Las Vegas, Nevada."
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=venue.source_urls,
        additional_instruction=(
            "Verify that the venue is located in Las Vegas, NV (the city or the Las Vegas Strip area). "
            "If an address lists 'Las Vegas, NV' or an adjacent unincorporated area commonly associated with the Strip (e.g., Paradise, NV), "
            "consider it acceptable for the purpose of this residency task."
        )
    )

    # 2) Entertainment_Venue_Type
    type_node = evaluator.add_leaf(
        id="Entertainment_Venue_Type",
        desc="The venue is established as a live entertainment/concert venue",
        parent=complete_node,
        critical=True
    )
    type_claim = (
        f"'{venue.official_name}' is a live entertainment or concert venue (e.g., a theater, showroom, or concert hall)."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=venue.source_urls,
        additional_instruction=(
            "Confirm from the cited page(s) that the venue hosts live performances such as concerts, residencies, or shows. "
            "Accept descriptions like 'theater', 'showroom', 'concert venue', or similar."
        )
    )

    # 3) Capacity_Within_Range
    capacity_range_node = evaluator.add_leaf(
        id="Capacity_Within_Range",
        desc="The venue's seating capacity is between 4,000 and 6,000 seats (inclusive)",
        parent=complete_node,
        critical=True
    )
    capacity_range_claim = (
        f"The seating capacity of '{venue.official_name}' is between 4,000 and 6,000 seats inclusive."
    )
    await evaluator.verify(
        claim=capacity_range_claim,
        node=capacity_range_node,
        sources=venue.source_urls,
        additional_instruction=(
            "Use the cited source(s) to check the venue's seating capacity. "
            "If the source gives an exact number (e.g., 4,100), verify it lies within 4,000–6,000. "
            "If the source provides a small variation (e.g., 'about 4,100', 'approximately 4,500'), "
            "still consider it within range if the implied capacity is between 4,000 and 6,000."
        )
    )

    # Add some custom info for transparency
    evaluator.add_custom_info(
        info={
            "selected_venue": {
                "official_name": venue.official_name,
                "seating_capacity_raw": venue.seating_capacity,
                "street_address_raw": venue.street_address,
                "source_urls": venue.source_urls
            },
            "address_format_check": {
                "looks_like_street_address": looks_like_address(venue.street_address)
            }
        },
        info_type="extraction_summary",
        info_name="selected_venue_extraction"
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
    Evaluate an answer for the Las Vegas residency venue capacity task.
    """
    # Initialize evaluator with PARALLEL aggregation at root
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

    # Extraction
    venues_extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Select first venue (as task requires one)
    primary_venue = select_primary_venue(venues_extraction)

    # Build verification tree and run checks
    await build_verification_tree_for_venue(evaluator, root, primary_venue)

    # Return summary
    return evaluator.get_summary()