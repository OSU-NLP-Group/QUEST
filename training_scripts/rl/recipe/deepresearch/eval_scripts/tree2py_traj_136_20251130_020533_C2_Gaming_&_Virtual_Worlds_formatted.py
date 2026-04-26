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
TASK_ID = "esports_arlington_venue_specs"
TASK_DESCRIPTION = """
Identify the esports stadium located in Arlington, Texas, United States that opened in November 2018. Provide the following specifications for this venue: (1) the total square footage of its adaptable space, and (2) the size measurement of its LED displays.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured extraction of the venue identification and specifications from the answer."""
    venue_name: Optional[str] = None
    location: Optional[str] = None  # e.g., "Arlington, Texas, United States" or "Arlington, TX"
    opening_date: Optional[str] = None  # e.g., "November 2018", "Nov 2018", "2018-11"
    venue_type: Optional[str] = None  # e.g., "esports stadium", "esports facility"
    adaptable_space_sqft: Optional[str] = None  # e.g., "100,000 sq ft", "100000 square feet"
    led_display_size: Optional[str] = None  # e.g., "85-foot LED wall", "85 ft LED screen"
    sources: List[str] = Field(default_factory=list)  # URLs cited in the answer that support the venue details


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    From the provided answer, extract information about the esports venue that matches ALL of the following constraints:
    - Located in Arlington, Texas, United States (Arlington, TX is acceptable)
    - Opened in November 2018
    - Is an esports stadium or esports facility

    If the answer mentions multiple venues, select the single venue that best matches ALL constraints above. If none fully match, select the venue most likely intended for Arlington, TX and still extract fields as present. If a field is not present, return null.

    Extract the following fields:
    1. venue_name: The name of the venue.
    2. location: The venue's location as a single string (e.g., "Arlington, Texas, United States" or "Arlington, TX").
    3. opening_date: The month and year the venue opened (e.g., "November 2018", "Nov 2018", "2018-11").
    4. venue_type: The type of venue (e.g., "esports stadium", "esports facility", "esports venue").
    5. adaptable_space_sqft: The total square footage of the venue's adaptable/flexible space exactly as stated (include units if provided).
    6. led_display_size: The size measurement of the venue's LED display(s) exactly as stated (include units if provided; e.g., "85-foot LED wall").
    7. sources: An array of all URLs mentioned in the answer that specifically relate to this venue or its specifications. Extract explicit URLs only (including protocol). Include official pages, Wikipedia entries, news releases, or any pages cited.

    Rules:
    - Do not invent or infer any information not explicitly present in the answer.
    - If an item is missing in the answer, set its value to null.
    - For 'sources', only include explicit URLs found in the answer (plain URLs or markdown links).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_venue_identification(
    evaluator: Evaluator,
    parent_node,
    info: VenueExtraction
) -> None:
    """
    Build and verify the venue identification subtree:
    - Existence of essential identification info
    - Location matches Arlington, Texas, United States
    - Opening date matches November 2018
    - Venue type is esports stadium/facility
    """
    # Parallel critical node for identification checks
    ident_node = evaluator.add_parallel(
        id="venue_identification",
        desc="Correctly identify the venue that matches all stated constraints (location, opening month/year, and venue type)",
        parent=parent_node,
        critical=True
    )

    # Critical existence/gating check: venue name and at least one source URL
    _has_name = bool(info.venue_name and info.venue_name.strip())
    _has_sources = bool(info.sources and len(info.sources) > 0)
    evaluator.add_custom_node(
        result=(_has_name and _has_sources),
        id="venue_required_info",
        desc="Venue name and at least one source URL are provided in the answer",
        parent=ident_node,
        critical=True
    )

    # 1) Location match
    loc_node = evaluator.add_leaf(
        id="location_match",
        desc="The identified venue is located in Arlington, Texas, United States",
        parent=ident_node,
        critical=True
    )
    venue_for_claim = info.venue_name or "the venue"
    claim_loc = f"The venue '{venue_for_claim}' is located in Arlington, Texas, United States."
    await evaluator.verify(
        claim=claim_loc,
        node=loc_node,
        sources=info.sources,
        additional_instruction=(
            "Check whether the cited webpage(s) explicitly place the venue in Arlington, Texas, United States. "
            "Minor format variations like 'Arlington, TX' or omission of 'United States' are acceptable, "
            "as long as it clearly refers to Arlington, Texas."
        )
    )

    # 2) Opening date match: November 2018
    open_node = evaluator.add_leaf(
        id="opening_date_match",
        desc="The identified venue opened in November 2018",
        parent=ident_node,
        critical=True
    )
    claim_open = f"The venue '{venue_for_claim}' opened in November 2018."
    await evaluator.verify(
        claim=claim_open,
        node=open_node,
        sources=info.sources,
        additional_instruction=(
            "Verify that the page(s) explicitly state that the venue opened in November 2018. "
            "Accept common variants such as 'Nov 2018' or 'Opened: November 2018'."
        )
    )

    # 3) Venue type match: esports stadium/facility
    type_node = evaluator.add_leaf(
        id="venue_type_match",
        desc="The identified venue is an esports stadium or esports facility",
        parent=ident_node,
        critical=True
    )
    claim_type = f"The venue '{venue_for_claim}' is an esports stadium or esports facility."
    await evaluator.verify(
        claim=claim_type,
        node=type_node,
        sources=info.sources,
        additional_instruction=(
            "Confirm the venue is described as an esports stadium, esports facility, or closely equivalent term "
            "(e.g., esports arena/venue). Minor wording variations are acceptable."
        )
    )


async def verify_technical_specifications(
    evaluator: Evaluator,
    parent_node,
    info: VenueExtraction
) -> None:
    """
    Build and verify the technical specifications subtree:
    - Square footage of adaptable space
    - LED display size
    """
    specs_node = evaluator.add_parallel(
        id="technical_specifications",
        desc="Provide the required venue specifications from the question",
        parent=parent_node,
        critical=True
    )

    # Square footage existence (critical)
    has_sqft = bool(info.adaptable_space_sqft and info.adaptable_space_sqft.strip())
    evaluator.add_custom_node(
        result=has_sqft,
        id="square_footage_provided",
        desc="The total square footage of the venue's adaptable space is provided in the answer",
        parent=specs_node,
        critical=True
    )

    # Square footage verification (critical)
    sqft_leaf = evaluator.add_leaf(
        id="square_footage",
        desc="The total square footage of the venue's adaptable space is provided",
        parent=specs_node,
        critical=True
    )
    sqft_val = info.adaptable_space_sqft or ""
    claim_sqft = f"The venue's adaptable (flexible) space totals {sqft_val}."
    await evaluator.verify(
        claim=claim_sqft,
        node=sqft_leaf,
        sources=info.sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly mention the total adaptable/flexible space figure. "
            "Accept synonyms like 'adaptable space', 'flexible space', or 'multi-use space'. "
            "Minor formatting differences (commas, hyphens, or 'square feet' vs 'sq ft') are acceptable."
        )
    )

    # LED display existence (critical)
    has_led = bool(info.led_display_size and info.led_display_size.strip())
    evaluator.add_custom_node(
        result=has_led,
        id="led_display_size_provided",
        desc="The size measurement of the venue's LED displays is provided in the answer",
        parent=specs_node,
        critical=True
    )

    # LED display verification (critical)
    led_leaf = evaluator.add_leaf(
        id="led_display_size",
        desc="The size measurement of the venue's LED displays is provided",
        parent=specs_node,
        critical=True
    )
    led_val = info.led_display_size or ""
    claim_led = f"The venue's LED display(s) measure {led_val}."
    await evaluator.verify(
        claim=claim_led,
        node=led_leaf,
        sources=info.sources,
        additional_instruction=(
            "Verify that the cited page(s) state the size of the venue's LED display(s). "
            "Accept terms like LED wall, LED screen, LED board, etc. "
            "Minor unit or punctuation variations are acceptable if the measurement clearly matches."
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
    Evaluate the agent's answer for the Arlington esports venue identification and specs task.
    Returns a standardized summary dict containing the verification tree and final score.
    """
    # Initialize evaluator with a sequential root (the overall task has sequential dependency: identify venue first, then specs)
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

    # Extract structured venue info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Create a critical sequential node under root to respect rubric's critical root nature
    task_root = evaluator.add_sequential(
        id="task_root",
        desc="Identify the correct esports venue in Arlington, TX that opened in November 2018 and provide the required specifications",
        parent=root,
        critical=True
    )

    # Build verification subtrees
    await verify_venue_identification(evaluator, task_root, extracted_info)
    await verify_technical_specifications(evaluator, task_root, extracted_info)

    # Return evaluation summary
    return evaluator.get_summary()