import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "austin_premium_office_building"
TASK_DESCRIPTION = """I am seeking to identify a premium office building in Austin, Texas, for our company's headquarters. The building must meet the following specific requirements:

1. Has achieved at least LEED Silver certification
2. Has a gross floor area of at least 150,000 square feet
3. Provides structured parking
4. Is classified as Class A office space
5. Is located in Austin, Texas
6. Has at least 15 stories
7. Was completed or underwent major renovation after 2015
8. Provides a parking ratio of at least 2.5 spaces per 1,000 square feet

What is the name and street address of an office building that satisfies all of these requirements?"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class BuildingExtraction(BaseModel):
    """Information about the single building proposed in the answer."""
    building_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    leed_level: Optional[str] = None
    gross_floor_area: Optional[str] = None
    structured_parking: Optional[str] = None  # e.g., "structured parking", "parking garage", Yes/No, description, etc.
    building_class: Optional[str] = None      # e.g., "Class A", "Class AA", "A+"
    stories: Optional[str] = None             # keep as string to be lenient (e.g., "17", "17 stories")
    completion_year: Optional[str] = None     # year string if present
    renovation_year: Optional[str] = None     # major renovation year string if present
    parking_ratio: Optional[str] = None       # e.g., "3.0/1000", "2.7 per 1,000 sf"
    source_urls: List[str] = Field(default_factory=list)  # all URLs explicitly cited in the answer for this building


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_building() -> str:
    return """
Extract the single office building proposed by the answer as the recommended option (if multiple buildings are mentioned, choose the first one that appears to be the main recommendation; otherwise pick the first building mentioned). Return the following fields exactly as they appear in the answer; do not infer or invent any values:

- building_name: the building's name
- street_address: the full street address if provided (include number, street, suite if any)
- city: the city name if mentioned
- state: the state if mentioned
- leed_level: the stated LEED certification level (e.g., "LEED Silver", "LEED Gold", "LEED Platinum"), if mentioned
- gross_floor_area: the stated gross floor area or building size (keep unit text, e.g., "200,000 SF"), if mentioned
- structured_parking: the stated parking type/availability (e.g., "structured parking", "parking garage", "podium garage"), or a yes/no statement if the answer explicitly says so
- building_class: the stated property class (e.g., "Class A", "Class AA", "A+"), if mentioned
- stories: the number of stories if mentioned (keep as provided, e.g., "17", "17 stories")
- completion_year: the completion year if mentioned (4-digit), else null
- renovation_year: a major renovation year if mentioned (4-digit), else null
- parking_ratio: the stated parking ratio if mentioned (e.g., "3.0/1,000 sf"), keep as provided
- source_urls: an array of all URLs explicitly provided in the answer that are intended to support the building and its attributes.
  IMPORTANT:
  • Extract only URLs explicitly present in the answer (including markdown links).
  • Include full URLs with protocol (prepend http:// if missing).
  • Do not include non-URL citations (e.g., "according to ...") unless an actual URL is present.

If any field is not present in the answer, set it to null (or [] for source_urls). Do NOT rely on external knowledge; use only the answer text.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(evaluator: Evaluator, parent, data: BuildingExtraction) -> None:
    """Create all rubric leaf nodes under the (parallel) root and run verifications."""

    # Convenience variables
    name = (data.building_name or "").strip()
    addr = (data.street_address or "").strip()
    srcs = data.source_urls or []

    # 1) Name provided (existence) - critical
    evaluator.add_custom_node(
        result=bool(name),
        id="building_name_provided",
        desc="The solution provides the name of the building",
        parent=parent,
        critical=True
    )

    # 2) Street address provided (existence) - critical
    evaluator.add_custom_node(
        result=bool(addr),
        id="street_address_provided",
        desc="The solution provides the street address of the building",
        parent=parent,
        critical=True
    )

    # 3) LEED certification (>= Silver) - critical
    node_leed = evaluator.add_leaf(
        id="leed_certification",
        desc="The building has achieved at least LEED Silver certification",
        parent=parent,
        critical=True
    )
    claim_leed = f"The building named '{name}' has achieved LEED Silver certification or higher (Gold/Platinum also qualify as meeting at least Silver)."
    await evaluator.verify(
        claim=claim_leed,
        node=node_leed,
        sources=srcs,
        additional_instruction=(
            "Verify that the provided source(s) explicitly indicate the building has achieved a LEED level that is "
            "Silver or above. Accept LEED Core & Shell or LEED O+M certifications if clearly applicable to the building."
        )
    )

    # 4) Building size >= 150,000 SF - critical
    node_size = evaluator.add_leaf(
        id="building_size",
        desc="The building has a gross floor area of at least 150,000 square feet",
        parent=parent,
        critical=True
    )
    claim_size = f"The building named '{name}' has a gross floor area (or total/rentable/usable building area) of at least 150,000 square feet."
    await evaluator.verify(
        claim=claim_size,
        node=node_size,
        sources=srcs,
        additional_instruction=(
            "Look for explicit figures for building size/area (e.g., GFA, GLA, NRA, RSF). If the value is clearly "
            ">= 150,000 SF, then the claim is supported. If multiple numbers are given, use the most appropriate total "
            "building area figure. Minor rounding is acceptable."
        )
    )

    # 5) Structured parking - critical
    node_parking_struct = evaluator.add_leaf(
        id="parking_availability",
        desc="The building provides structured parking",
        parent=parent,
        critical=True
    )
    claim_structured = f"The building named '{name}' provides structured parking (e.g., a parking garage or podium/structured facility)."
    await evaluator.verify(
        claim=claim_structured,
        node=node_parking_struct,
        sources=srcs,
        additional_instruction=(
            "Confirm that the parking is structured (e.g., garage, podium, deck). Surface-only parking alone does not qualify."
        )
    )

    # 6) Building classified as Class A - critical
    node_class = evaluator.add_leaf(
        id="building_class",
        desc="The building is classified as Class A office space",
        parent=parent,
        critical=True
    )
    claim_class = f"The building named '{name}' is categorized as Class A office space."
    await evaluator.verify(
        claim=claim_class,
        node=node_class,
        sources=srcs,
        additional_instruction=(
            "Accept Class A, A+, or AA classifications. Confirmation from brokers, property managers, or official property "
            "materials is acceptable."
        )
    )

    # 7) Located in Austin, Texas - critical
    node_location = evaluator.add_leaf(
        id="austin_location",
        desc="The building is located in Austin, Texas",
        parent=parent,
        critical=True
    )
    claim_location = f"The building named '{name}' is located in Austin, Texas."
    await evaluator.verify(
        claim=claim_location,
        node=node_location,
        sources=srcs,
        additional_instruction=(
            "Verify city and state on the property or reputable listing page. 'Austin, TX' or 'Austin, Texas' qualifies. "
            f"If the address was provided in the answer ({addr}), you may use it for cross-checking."
        )
    )

    # 8) At least 15 stories - critical
    node_height = evaluator.add_leaf(
        id="building_height",
        desc="The building has at least 15 stories",
        parent=parent,
        critical=True
    )
    claim_height = f"The building named '{name}' has at least 15 stories (floors)."
    await evaluator.verify(
        claim=claim_height,
        node=node_height,
        sources=srcs,
        additional_instruction="Confirm the stated number of floors or stories is >= 15."
    )

    # 9) Completed or major renovation after 2015 - critical
    node_recent = evaluator.add_leaf(
        id="recent_completion",
        desc="The building was completed or underwent major renovation after 2015",
        parent=parent,
        critical=True
    )
    claim_recent = f"The building named '{name}' was completed after 2015 or underwent a major renovation after 2015."
    await evaluator.verify(
        claim=claim_recent,
        node=node_recent,
        sources=srcs,
        additional_instruction=(
            "Validate that either (a) the completion year is >= 2016, or (b) a major renovation year is >= 2016. "
            "If both are present, either satisfying the condition is acceptable."
        )
    )

    # 10) Parking ratio >= 2.5 spaces per 1,000 SF - critical
    node_ratio = evaluator.add_leaf(
        id="parking_ratio",
        desc="The building provides a parking ratio of at least 2.5 spaces per 1,000 square feet",
        parent=parent,
        critical=True
    )
    claim_ratio = f"The building named '{name}' has a parking ratio of at least 2.5 spaces per 1,000 square feet."
    await evaluator.verify(
        claim=claim_ratio,
        node=node_ratio,
        sources=srcs,
        additional_instruction=(
            "Look for a stated parking ratio (e.g., '2.5/1,000 SF', '3.0 per 1,000 square feet'). "
            "If the ratio is >= 2.5, the claim is supported. Minor rounding is acceptable."
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
    Evaluate an answer for the Austin premium office building identification task.
    """
    # Initialize evaluator with parallel root (as per rubric)
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

    # Extract building details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_building(),
        template_class=BuildingExtraction,
        extraction_name="building_extraction"
    )

    # Optional: record simple summary for convenience
    evaluator.add_custom_info(
        info={
            "building_name": extracted.building_name,
            "street_address": extracted.street_address,
            "city": extracted.city,
            "state": extracted.state,
            "leed_level": extracted.leed_level,
            "gross_floor_area": extracted.gross_floor_area,
            "structured_parking": extracted.structured_parking,
            "building_class": extracted.building_class,
            "stories": extracted.stories,
            "completion_year": extracted.completion_year,
            "renovation_year": extracted.renovation_year,
            "parking_ratio": extracted.parking_ratio,
            "source_urls_count": len(extracted.source_urls or []),
        },
        info_type="extracted_overview"
    )

    # Build all rubric leaves and verify
    await build_and_verify_nodes(evaluator, root, extracted)

    # Return structured result
    return evaluator.get_summary()