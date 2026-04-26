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
TASK_ID = "indoor_nv_venue"
TASK_DESCRIPTION = (
    "Identify an indoor concert venue located in Nevada, United States, that meets the following specifications: "
    "(1) The venue must have a seating capacity between 5,000 and 10,000 seats for concert events. "
    "(2) The venue must be an indoor theater or arena (not an outdoor amphitheater). "
    "(3) The venue must be actively hosting live music concerts or residency shows in 2024-2025. "
    "(4) The venue must comply with ADA accessibility requirements by providing wheelchair accessible seating comprising at least 1% of total capacity with companion seats. "
    "(5) The venue must have at least 4 emergency exits or exit access doorways to comply with safety regulations for large gatherings. "
    "Provide the name of the venue, its specific seating capacity for concerts, its exact location (city and state), "
    "and reference URLs that verify each of the above requirements."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    concert_seating_capacity: Optional[str] = None  # keep as string for robustness (e.g., "7,000" or "5,500–7,000")
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    From the answer, extract the following fields exactly as presented:
    - venue_name: The name of the venue identified.
    - concert_seating_capacity: The specific seating capacity figure for concert events (string; do not parse to number).
    - location_city: The city of the venue.
    - location_state: The state of the venue (e.g., NV or Nevada).
    - reference_urls: A list of all explicit URLs the answer provides as references/sources. Include any URL format: raw URL, markdown link targets, or any listed reference links. Deduplicate if repeated.

    Rules:
    - If a field is missing, set it to null (for strings) or [] (for lists).
    - Extract URLs only if explicitly present in the answer.
    - Do not invent or infer any values.
    """


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_required_output_fields(
    evaluator: Evaluator,
    parent_node,
    data: VenueExtraction
) -> None:
    """
    Build and evaluate the 'Response_Provides_Required_Output_Fields' subtree.
    All children are critical because the task root node is critical.
    """
    req_node = evaluator.add_parallel(
        id="Response_Provides_Required_Output_Fields",
        desc="Check that the response includes all required output fields (not whether they are correct).",
        parent=parent_node,
        critical=True
    )

    # Venue name provided
    evaluator.add_custom_node(
        result=bool(data.venue_name and data.venue_name.strip()),
        id="Venue_Name_Provided",
        desc="Provides the name of the venue.",
        parent=req_node,
        critical=True
    )

    # Concert seating capacity provided (string is fine)
    evaluator.add_custom_node(
        result=bool(data.concert_seating_capacity and data.concert_seating_capacity.strip()),
        id="Concert_Seating_Capacity_Provided",
        desc="Provides the venue's specific seating capacity for concert events (a concrete value or clearly stated concert capacity figure).",
        parent=req_node,
        critical=True
    )

    # Location city and state provided
    evaluator.add_custom_node(
        result=bool(data.location_city and data.location_city.strip()) and bool(data.location_state and data.location_state.strip()),
        id="Location_City_and_State_Provided",
        desc="Provides the venue's exact location as city and state.",
        parent=req_node,
        critical=True
    )

    # Reference URLs provided (at least one) — this only checks presence; verification happens below
    evaluator.add_custom_node(
        result=bool(data.reference_urls and len(data.reference_urls) > 0),
        id="Reference_URLs_Provided_and_Verify_Claims",
        desc="Provides reference URL(s) that collectively verify each required constraint (capacity range, Nevada location, indoor type, active 2024-2025 hosting, ADA wheelchair/companion seating, and ≥4 emergency exits).",
        parent=req_node,
        critical=True
    )


async def build_constraints_checks(
    evaluator: Evaluator,
    parent_node,
    data: VenueExtraction
) -> None:
    """
    Build and evaluate the 'Venue_Meets_All_Constraints' subtree.
    Each child is a leaf verification using the provided sources.
    """
    constraints_node = evaluator.add_parallel(
        id="Venue_Meets_All_Constraints",
        desc="Check that the identified venue satisfies each constraint stated in the proposed question.",
        parent=parent_node,
        critical=True
    )

    # Helper vars
    name = data.venue_name or "the venue"
    city = data.location_city or "the stated city"
    state = data.location_state or "Nevada"
    sources = data.reference_urls

    # 1) Capacity between 5,000 and 10,000 (inclusive) for concerts
    cap_node = evaluator.add_leaf(
        id="Capacity_Between_5000_and_10000",
        desc="The venue's concert seating capacity is between 5,000 and 10,000 seats (inclusive, as stated).",
        parent=constraints_node,
        critical=True
    )
    cap_claim = f"{name} has a concert seating capacity between 5,000 and 10,000 seats (inclusive)."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=sources,
        additional_instruction=(
            "Verify the seating capacity specifically for concerts (not maximum occupancy or standing-room-only unless the page explicitly calls it 'concert capacity'). "
            "Allow reasonable textual variations like 'about', 'up to', or ranges so long as the capacity for concerts clearly falls within 5,000–10,000 inclusive. "
            "If multiple capacities are listed for different configurations, look for the concert configuration."
        ),
    )

    # 2) Located in Nevada, USA
    loc_node = evaluator.add_leaf(
        id="Located_in_Nevada_USA",
        desc="The venue is located in Nevada, United States.",
        parent=constraints_node,
        critical=True
    )
    # Keep the claim focused on Nevada to avoid false negatives from city name variants
    loc_claim = f"{name} is located in Nevada, United States."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the venue is in the state of Nevada (NV). "
            "If the source uses 'NV' abbreviation, treat it as equivalent to 'Nevada'. "
            "Do not rely on generic company or promoter location; it must be the venue's location."
        ),
    )

    # 3) Indoor theater or arena (not outdoor amphitheater)
    indoor_node = evaluator.add_leaf(
        id="Indoor_Theater_or_Arena",
        desc="The venue is an indoor theater or arena (not an outdoor amphitheater).",
        parent=constraints_node,
        critical=True
    )
    indoor_claim = f"{name} is an indoor theater or an indoor arena (not an outdoor amphitheater)."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit indications that the venue is an indoor facility (e.g., enclosed arena, theater). "
            "If sources describe the venue as an amphitheater or outdoor, the claim is not supported. "
            "Use both text and screenshots (photos of interior, roof, etc.) if needed."
        ),
    )

    # 4) Actively hosting live music concerts or residencies in 2024–2025
    active_node = evaluator.add_leaf(
        id="Actively_Hosting_in_2024_2025",
        desc="The venue is actively hosting live music concerts or residency shows in 2024-2025.",
        parent=constraints_node,
        critical=True
    )
    active_claim = f"{name} hosted or scheduled live music concerts or artist residencies in 2024 or 2025."
    await evaluator.verify(
        claim=active_claim,
        node=active_node,
        sources=sources,
        additional_instruction=(
            "Check event calendars, news releases, or schedule pages for years 2024 or 2025 that show concerts or residency shows. "
            "References to older years do not suffice. Evidence of scheduled or completed 2024/2025 shows is acceptable."
        ),
    )

    # 5) ADA wheelchair-accessible seating ≥ 1% with companion seats
    ada_node = evaluator.add_leaf(
        id="ADA_Wheelchair_Seating_At_Least_1pct_with_Companion",
        desc="The venue provides wheelchair-accessible seating comprising at least 1% of total capacity, with companion seats.",
        parent=constraints_node,
        critical=True
    )
    ada_claim = (
        f"{name} provides wheelchair-accessible seating comprising at least 1% of total capacity and provides companion seats."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit statements or policies indicating that accessible seating equals or exceeds 1% of total capacity and that companion seating is provided. "
            "A generic 'ADA compliant' statement alone is insufficient unless it explicitly notes ≥1% accessible seating and companion seats."
        ),
    )

    # 6) At least 4 emergency exits / exit access doorways
    exits_node = evaluator.add_leaf(
        id="At_Least_4_Emergency_Exits",
        desc="The venue has at least 4 emergency exits or exit access doorways.",
        parent=constraints_node,
        critical=True
    )
    exits_claim = f"{name} has at least four emergency exits or exit access doorways."
    await evaluator.verify(
        claim=exits_claim,
        node=exits_node,
        sources=sources,
        additional_instruction=(
            "Seek floor plans, safety plans, or official documentation that indicates the number of exits/egress doors. "
            "Accept synonyms such as exit doors, emergency egress, or exit access doorways. "
            "If the sources do not state or show a count ≥ 4, the claim is not supported."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Entry point for evaluating an answer for the Indoor Nevada Concert Venue task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Align with rubric root being sequential
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

    # Top-level critical sequential node reflecting the rubric root
    task_node = evaluator.add_sequential(
        id="Indoor_Concert_Venue_in_Nevada_Task",
        desc="Evaluate whether the response identifies a single indoor Nevada concert venue meeting all constraints and provides the requested details with verifying sources.",
        parent=root,
        critical=True
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_info"
    )

    # Build and evaluate subtrees
    await build_required_output_fields(evaluator, task_node, extracted)
    await build_constraints_checks(evaluator, task_node, extracted)

    return evaluator.get_summary()