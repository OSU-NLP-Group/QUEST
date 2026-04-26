import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "msg_capacity_infrastructure"
TASK_DESCRIPTION = """
What is the concert capacity of Madison Square Garden in New York City, and does it have built-in audio/video/lighting systems?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None


class CapacityInfo(BaseModel):
    capacity_text: Optional[str] = None  # The capacity wording as stated in the answer (e.g., "around 20,000")
    capacity_number: Optional[str] = None  # If a single number appears (keep as string to avoid parsing issues)
    sources: List[str] = Field(default_factory=list)  # Any URLs cited in the answer related to capacity


class InfrastructureInfo(BaseModel):
    has_built_in_audio: Optional[bool] = None
    has_built_in_video: Optional[bool] = None
    has_built_in_lighting: Optional[bool] = None
    statement_text: Optional[str] = None  # The exact sentence or phrase in the answer describing built-in systems
    sources: List[str] = Field(default_factory=list)  # Any URLs cited in the answer related to infrastructure


class MSGExtraction(BaseModel):
    venue: Optional[VenueInfo] = None
    capacity: Optional[CapacityInfo] = None
    infrastructure: Optional[InfrastructureInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_msg_structured() -> str:
    return """
    Extract exactly and only the information explicitly stated in the answer about Madison Square Garden (MSG).

    Return a JSON object with the following structure:
    {
      "venue": {
        "venue_name": string | null,
        "city": string | null,
        "state": string | null,
        "country": string | null
      },
      "capacity": {
        "capacity_text": string | null,
        "capacity_number": string | null,
        "sources": string[]    // URLs explicitly mentioned in the answer that relate to capacity
      },
      "infrastructure": {
        "has_built_in_audio": boolean | null,
        "has_built_in_video": boolean | null,
        "has_built_in_lighting": boolean | null,
        "statement_text": string | null,
        "sources": string[]    // URLs explicitly mentioned in the answer that relate to audio/video/lighting
      }
    }

    Detailed rules:
    - venue_name and city/state/country: extract exactly what the answer states for the venue and its location.
      If the answer uses a shorthand like "MSG", still extract it as the venue_name.
    - capacity_text: the full phrase describing concert capacity (e.g., "about 20,000", "up to 19,500").
      capacity_number: extract a single numeric figure only if the answer clearly provides one number; otherwise null.
    - For sources arrays, include only actual URLs found in the answer (plain URLs or markdown links). Do not invent.
    - For infrastructure booleans, set to true/false only if the answer explicitly states presence/absence of built-in
      audio, video, or lighting systems. If the answer is ambiguous or does not mention a category, set it to null.
    - statement_text should capture the exact sentence or phrase that discusses built-in audio/video/lighting systems.
    - If a field is not present in the answer, return null (or [] for arrays).
    """


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_msg_verification_tree(evaluator: Evaluator, extracted: MSGExtraction, parent_node) -> None:
    """
    Build the verification tree according to the rubric.
    Root (created by evaluator.initialize) -> critical parallel node (MSG_...) -> 3 critical leaf checks.
    """
    # Create the main critical parallel node
    main_node = evaluator.add_parallel(
        id="MSG_Concert_Capacity_And_Infrastructure_Check",
        desc="Evaluate whether the answer addresses Madison Square Garden (NYC), provides its concert capacity meeting the minimum, and states whether it has built-in audio/video/lighting systems.",
        parent=parent_node,
        critical=True
    )

    # Prepare extracted fields for use in additional instructions
    vn = (extracted.venue.venue_name if extracted and extracted.venue else None) or "null"
    city = (extracted.venue.city if extracted and extracted.venue else None) or "null"
    state = (extracted.venue.state if extracted and extracted.venue else None) or "null"
    country = (extracted.venue.country if extracted and extracted.venue else None) or "null"

    cap_text = (extracted.capacity.capacity_text if extracted and extracted.capacity else None) or "null"
    cap_num = (extracted.capacity.capacity_number if extracted and extracted.capacity else None) or "null"

    infra_stmt = (extracted.infrastructure.statement_text if extracted and extracted.infrastructure else None) or "null"
    bina = (extracted.infrastructure.has_built_in_audio if extracted and extracted.infrastructure else None)
    binv = (extracted.infrastructure.has_built_in_video if extracted and extracted.infrastructure else None)
    binl = (extracted.infrastructure.has_built_in_lighting if extracted and extracted.infrastructure else None)

    bina_str = "true" if bina is True else ("false" if bina is False else "null")
    binv_str = "true" if binv is True else ("false" if binv is False else "null")
    binl_str = "true" if binl is True else ("false" if binl is False else "null")

    # 1) Venue identity and location leaf
    node_venue = evaluator.add_leaf(
        id="Venue_Identity_And_Location",
        desc="Answer identifies the venue as Madison Square Garden located in New York City.",
        parent=main_node,
        critical=True
    )
    claim_venue = "The answer identifies the venue as Madison Square Garden located in New York City."
    add_ins_venue = (
        f"Use the answer text only. The extracted fields are: venue_name='{vn}', city='{city}', state='{state}', country='{country}'. "
        "Accept reasonable variants such as 'MSG' for the venue and 'NYC', 'New York, NY', or 'Manhattan, New York City' for the location. "
        "The claim is correct only if the answer clearly refers to Madison Square Garden and places it in New York City."
    )

    # 2) Concert capacity minimum leaf
    node_capacity = evaluator.add_leaf(
        id="Concert_Capacity_Minimum",
        desc="Answer states Madison Square Garden's concert capacity and the stated capacity is at least 19,000 people.",
        parent=main_node,
        critical=True
    )
    claim_capacity = "The answer states Madison Square Garden's concert capacity, and the stated capacity is at least 19,000 people."
    add_ins_capacity = (
        f"Evaluate based on the answer text. Extracted capacity_text='{cap_text}', capacity_number='{cap_num}'. "
        "If the answer provides a range or approximate phrasing (e.g., 'around 20,000', 'up to 19,500'), interpret reasonably. "
        "For ranges, the typical/maximum concert configuration capacity can be considered; however, the judgment must rely on the answer's stated capacity. "
        "If the answer does not state any concert capacity, or if it states a capacity below 19,000, mark the claim incorrect."
    )

    # 3) Built-in audio/video/lighting systems leaf
    node_infra = evaluator.add_leaf(
        id="Built_In_AV_Lighting_Systems",
        desc="Answer states whether Madison Square Garden has built-in audio/video/lighting systems.",
        parent=main_node,
        critical=True
    )
    claim_infra = "The answer explicitly states whether Madison Square Garden has built-in audio, video, and lighting systems."
    add_ins_infra = (
        f"Evaluate based on the answer text. Extracted statement_text='{infra_stmt}', "
        f"has_built_in_audio={bina_str}, has_built_in_video={binv_str}, has_built_in_lighting={binl_str}. "
        "It is sufficient if the answer clearly indicates presence or absence of built-in (in-house) audio, video, and lighting infrastructure—either individually or collectively. "
        "If the answer is silent or ambiguous about these built-in systems, mark the claim incorrect."
    )

    # Run the three verifications in parallel
    await evaluator.batch_verify([
        (claim_venue, None, node_venue, add_ins_venue),
        (claim_capacity, None, node_capacity, add_ins_capacity),
        (claim_infra, None, node_infra, add_ins_infra),
    ])


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
    Evaluate an answer for Madison Square Garden concert capacity and infrastructure.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Children checks are independent
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_msg_structured(),
        template_class=MSGExtraction,
        extraction_name="msg_structured_extraction",
    )

    # Optionally record the extracted values for transparency
    evaluator.add_custom_info(
        info={
            "venue": extracted.venue.dict() if extracted.venue else {},
            "capacity": extracted.capacity.dict() if extracted.capacity else {},
            "infrastructure": extracted.infrastructure.dict() if extracted.infrastructure else {},
        },
        info_type="extracted_fields",
        info_name="extracted_msg_fields",
    )

    # Build and run verification checks
    await build_msg_verification_tree(evaluator, extracted, root)

    # Return structured evaluation summary
    return evaluator.get_summary()