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
TASK_ID = "us_esports_venue_requirements"
TASK_DESCRIPTION = (
    "Identify a purpose-built esports venue in the United States that meets the following specifications: "
    "it must have a seating capacity of at least 2,000, a total venue space of at least 80,000 square feet, "
    "and feature large-format LED display technology. Provide the following information about this venue: "
    "(1) The official name of the venue, (2) The city and state where it is located, (3) The exact seating capacity, "
    "(4) The total square footage of the facility, (5) Specifications of the LED display system (including size/dimensions), "
    "(6) Evidence that the venue is purpose-built or specifically designed for esports events, and (7) At least one authoritative "
    "reference URL (from the venue's official website or a reputable source) that verifies this information."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as free text for robustness (e.g., "2,500", "2,000+")
    total_square_footage: Optional[str] = None  # Free text (e.g., "80,000 sq ft", "90,000 square feet")
    led_display_spec: Optional[str] = None  # Technology/spec summary (e.g., "LED wall, 60ft x 20ft, 4mm pixel pitch")
    led_display_dimensions: Optional[str] = None  # Specific size/dimensions if present
    purpose_built_evidence: Optional[str] = None  # Textual evidence/statement from the answer (e.g., "purpose-built for esports")
    reference_urls: List[str] = Field(default_factory=list)  # All authoritative or official URLs provided in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract details for a single purpose-built esports venue mentioned in the answer. If multiple venues are mentioned,
    choose the first one that the answer appears to focus on. Return the following fields exactly as presented in the answer:
    1) venue_name: Official name of the venue.
    2) city: The city where the venue is located.
    3) state: The U.S. state where the venue is located (use the answer's format, e.g., "NV" or "Nevada").
    4) capacity: The seating capacity as stated in the answer (keep original formatting; do not convert to numbers).
    5) total_square_footage: The total venue space as stated (keep units and formatting).
    6) led_display_spec: Any description/specification of the LED display technology (e.g., type, vendor, pixel pitch).
    7) led_display_dimensions: Size or dimensions of the LED display (e.g., "60 ft x 20 ft"; return null if not specified).
    8) purpose_built_evidence: A phrase/sentence in the answer that indicates the venue is purpose-built for esports (return null if not present).
    9) reference_urls: Extract all URLs explicitly listed in the answer that serve as authoritative or official references (official venue site,
       major reputable publications, or vendor/installation case studies). Return valid full URLs only; if none are present, return an empty list.
    If any field is missing in the answer, return null for that field (or an empty list for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _choose_led_dim_text(extracted: VenueExtraction) -> Optional[str]:
    return extracted.led_display_dimensions if _non_empty(extracted.led_display_dimensions) else extracted.led_display_spec


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: VenueExtraction) -> None:
    """
    Construct the verification tree according to the rubric, separating existence checks and source-supported checks
    for clarity and robust gating. The top-level node is critical and aggregates all child checks in parallel.
    """
    # Create a critical aggregation node under the framework root
    task_root = evaluator.add_parallel(
        id="esports_venue_root",
        desc="The identified venue meets all specified requirements for hosting a large-scale esports tournament",
        parent=evaluator.root,
        critical=True
    )

    # Build a reference URL gate first (critical) so other verifications can depend on it
    ref_node = evaluator.add_parallel(
        id="reference_url",
        desc="A valid reference URL from an official or authoritative source is provided to verify the venue information",
        parent=task_root,
        critical=True
    )

    ref_exists = evaluator.add_custom_node(
        result=(bool(extracted.reference_urls) and len(extracted.reference_urls) > 0),
        id="reference_url_exists",
        desc="At least one reference URL is provided in the answer",
        parent=ref_node,
        critical=True
    )

    ref_auth_leaf = evaluator.add_leaf(
        id="reference_url_authoritative",
        desc="At least one provided URL is an official or authoritative source for the venue",
        parent=ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is an official or authoritative source for the venue.",
        node=ref_auth_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Judge authority by domain/content: official venue websites, major reputable publications, large venue operators, "
            "or vendor case studies explicitly documenting the installation at this venue count as authoritative. "
            "If none of the provided URLs meet this standard, return not supported."
        )
    )

    # Venue name checks (critical)
    name_node = evaluator.add_parallel(
        id="venue_name",
        desc="The specific name of the venue is provided",
        parent=task_root,
        critical=True
    )
    name_exists = evaluator.add_custom_node(
        result=_non_empty(extracted.venue_name),
        id="venue_name_provided",
        desc="Venue name is provided in the answer",
        parent=name_node,
        critical=True
    )
    name_supported = evaluator.add_leaf(
        id="venue_name_supported",
        desc="The official name of the venue is supported by the cited sources",
        parent=name_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the venue is '{extracted.venue_name or ''}'.",
        node=name_supported,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Verify the venue's official name exactly or with minor acceptable variations (hyphens, punctuation, abbreviations). "
            "Only pass if the source explicitly confirms this name."
        ),
        extra_prerequisites=[ref_exists]
    )

    # City and state checks (critical)
    city_node = evaluator.add_parallel(
        id="city_location",
        desc="The city and state where the venue is located are identified",
        parent=task_root,
        critical=True
    )
    city_exists = evaluator.add_custom_node(
        result=_non_empty(extracted.city) and _non_empty(extracted.state),
        id="city_state_provided",
        desc="Both city and state are provided in the answer",
        parent=city_node,
        critical=True
    )
    city_supported = evaluator.add_leaf(
        id="city_state_supported",
        desc="The venue location (city and state) is supported by the cited sources",
        parent=city_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue is located in {extracted.city or ''}, {extracted.state or ''}.",
        node=city_supported,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Accept common state abbreviations (e.g., NV for Nevada). Only pass if the source clearly indicates the venue is in the stated city and state."
        ),
        extra_prerequisites=[ref_exists]
    )

    # United States check (critical, direct leaf under root)
    us_leaf = evaluator.add_leaf(
        id="location_us",
        desc="The venue is located in the United States",
        parent=task_root,
        critical=True
    )
    await evaluator.verify(
        claim="This venue is located in the United States.",
        node=us_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Determine U.S. location from explicit mentions (e.g., 'USA', 'United States') or by recognizing U.S. cities/states. "
            "If the source indicates a non-U.S. location, fail."
        ),
        extra_prerequisites=[ref_exists]
    )

    # Capacity specification (critical)
    cap_node = evaluator.add_parallel(
        id="capacity_specification",
        desc="The venue's seating capacity is provided and is at least 2,000 seats",
        parent=task_root,
        critical=True
    )
    cap_exists = evaluator.add_custom_node(
        result=_non_empty(extracted.capacity),
        id="capacity_provided",
        desc="Seating capacity is provided in the answer",
        parent=cap_node,
        critical=True
    )
    cap_threshold = evaluator.add_leaf(
        id="capacity_at_least_2000",
        desc="The venue's seating capacity is at least 2,000 seats (source-supported)",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue's seating capacity is at least 2,000 seats; the answer claims '{extracted.capacity or ''}'.",
        node=cap_threshold,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Check the source for capacity. Pass if capacity stated is 2,000 or higher (allow reasonable rounding or '2,000+' notation)."
        ),
        extra_prerequisites=[ref_exists]
    )

    # Space specification (critical)
    space_node = evaluator.add_parallel(
        id="space_specification",
        desc="The total venue space is provided and is at least 80,000 square feet",
        parent=task_root,
        critical=True
    )
    space_exists = evaluator.add_custom_node(
        result=_non_empty(extracted.total_square_footage),
        id="space_provided",
        desc="Total square footage is provided in the answer",
        parent=space_node,
        critical=True
    )
    space_threshold = evaluator.add_leaf(
        id="space_at_least_80000",
        desc="The total venue space is at least 80,000 square feet (source-supported)",
        parent=space_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue has at least 80,000 square feet of total space; the answer claims '{extracted.total_square_footage or ''}'.",
        node=space_threshold,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Verify total venue space from the source. Pass if the figure is ≥ 80,000 sq ft (allow exact or approximate statements)."
        ),
        extra_prerequisites=[ref_exists]
    )

    # LED display technology (critical)
    led_node = evaluator.add_parallel(
        id="led_display",
        desc="The venue features large-format LED display technology with specifications provided",
        parent=task_root,
        critical=True
    )
    led_specs_provided = evaluator.add_custom_node(
        result=_non_empty(extracted.led_display_spec) or _non_empty(extracted.led_display_dimensions),
        id="led_specs_provided",
        desc="LED display specifications (including size/dimensions) are provided in the answer",
        parent=led_node,
        critical=True
    )
    led_presence_leaf = evaluator.add_leaf(
        id="led_presence_supported",
        desc="The venue features large-format LED display technology (source-supported)",
        parent=led_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue features large-format LED display technology (e.g., LED walls, large LED boards/screens).",
        node=led_presence_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm via source that large-format LED display technology is present (e.g., LED video wall installations). "
            "Vendor case studies documenting the specific venue are acceptable."
        ),
        extra_prerequisites=[ref_exists]
    )
    # Dimensions/specs support
    led_dims_text = _choose_led_dim_text(extracted) or ""
    led_dims_leaf = evaluator.add_leaf(
        id="led_dimensions_supported",
        desc="LED display size/dimensions/specifications are supported by the cited sources",
        parent=led_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LED display specifications include size/dimensions: '{led_dims_text}'.",
        node=led_dims_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Pass only if the source provides explicit LED size/dimensions or equivalent spec details "
            "(e.g., width/height in feet/meters, pixel pitch, resolution, or similar). Minor formatting variations are acceptable."
        ),
        extra_prerequisites=[ref_exists]
    )

    # Purpose-built for esports (critical)
    purpose_node = evaluator.add_parallel(
        id="purpose_built",
        desc="Evidence is provided that the venue is purpose-built or specifically designed for esports events",
        parent=task_root,
        critical=True
    )
    purpose_exists = evaluator.add_custom_node(
        result=_non_empty(extracted.purpose_built_evidence),
        id="purpose_built_evidence_provided",
        desc="The answer includes a statement indicating purpose-built esports design",
        parent=purpose_node,
        critical=True
    )
    purpose_supported = evaluator.add_leaf(
        id="purpose_built_supported",
        desc="The venue is purpose-built or specifically designed for esports (source-supported)",
        parent=purpose_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is purpose-built or specifically designed for esports events.",
        node=purpose_supported,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm from the source that the venue was built or designed expressly for esports (e.g., 'purpose-built for esports', "
            "'dedicated esports arena'). Strong evidence from the official site or reputable publications is expected."
        ),
        extra_prerequisites=[ref_exists]
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the US esports venue requirements task.
    """
    # Initialize evaluator with parallel root (we will add our own critical top-level node)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured venue info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()