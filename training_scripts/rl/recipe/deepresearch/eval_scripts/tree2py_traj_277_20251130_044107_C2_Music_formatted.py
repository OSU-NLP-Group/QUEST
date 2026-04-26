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
TASK_ID = "indoor_arena_sb2026"
TASK_DESCRIPTION = (
    "A music promoter is planning a major concert event to coincide with Super Bowl week in February 2026. "
    "Due to weather considerations, they require an indoor arena venue. For logistical reasons, the venue must be located "
    "within 5 miles of the Super Bowl 2026 host stadium. Among all indoor arenas that meet this location requirement, "
    "identify the one with the highest concert capacity. Provide the full name of the arena, its concert capacity, "
    "and the city where it is located."
)

REFERENCE_STADIUM_NAME = "Levi's Stadium"
REFERENCE_STADIUM_CITY = "Santa Clara, CA"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArenaExtraction(BaseModel):
    """
    Extract the single selected indoor arena claimed by the answer as the best choice,
    along with the requested output fields and the URL sources that support each constraint.
    """
    arena_full_name: Optional[str] = None
    concert_capacity: Optional[str] = None
    city: Optional[str] = None

    # Source URLs as cited in the answer text for each specific verification
    indoor_type_sources: List[str] = Field(default_factory=list)
    distance_sources: List[str] = Field(default_factory=list)
    capacity_sources: List[str] = Field(default_factory=list)
    highest_capacity_sources: List[str] = Field(default_factory=list)

    # Any other URLs the answer cites about the selected arena
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_arena_info() -> str:
    return """
    You must extract the single selected indoor arena that the answer claims is the best choice for the promoter's needs.

    Return a JSON object with the following fields:
    - arena_full_name: The full official name of the selected arena (string).
    - concert_capacity: The concert capacity of that arena as stated in the answer (string; keep any formatting as-is).
    - city: The city where that arena is located (string).
    - indoor_type_sources: An array of URLs cited in the answer that support the statement that this venue is an indoor arena. If none are provided, return an empty array.
    - distance_sources: An array of URLs cited in the answer that support that the venue is within 5 miles of Levi's Stadium in Santa Clara (e.g., a Google Maps directions link that shows the distance). If none are provided, return an empty array.
    - capacity_sources: An array of URLs cited in the answer that support the stated concert capacity for this arena. If none, return an empty array.
    - highest_capacity_sources: An array of URLs cited in the answer that support the claim that, among indoor arenas within 5 miles of Levi's Stadium, this arena has the highest concert capacity. This could include lists or comparisons of multiple nearby arenas. If none are given, return an empty array.
    - general_sources: Any other URLs in the answer specifically about this selected arena (official site, Wikipedia, venue pages, etc.). If none, return an empty array.

    IMPORTANT FOR URL EXTRACTION:
    - Only include URLs explicitly present in the answer. Do not invent or infer URLs.
    - Accept plain links or markdown links; extract the actual URL part.
    - Include full URLs with protocol. If missing protocol, prepend http://
    - If the answer mentions sources without URLs (e.g., "according to Wikipedia") but does not include the link, do not add a URL; just leave the relevant array empty.

    If any main field (arena_full_name, concert_capacity, city) is not present in the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists, filter out falsy or malformed items, and deduplicate preserving order."""
    merged: List[str] = []
    seen = set()
    for lst in lists:
        for url in lst or []:
            if not url or not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ArenaExtraction):
    """
    Build the verification tree and run all checks based on the rubric.
    """

    # Task node (critical)
    task_node = evaluator.add_parallel(
        id="Arena_Identification_Task",
        desc="Identify the indoor arena within 5 miles of Levi's Stadium with the highest concert capacity, and provide the requested arena details.",
        critical=True
    )

    # Node: Required Output Details (critical)
    required_output_node = evaluator.add_parallel(
        id="Required_Output_Details",
        desc="The answer provides all requested details for the selected arena.",
        parent=task_node,
        critical=True
    )

    # Existence checks for required fields (all critical leaves)
    name_provided = evaluator.add_custom_node(
        result=bool(extracted.arena_full_name and extracted.arena_full_name.strip()),
        id="Arena_Full_Name_Provided",
        desc="The full name of the selected arena is provided.",
        parent=required_output_node,
        critical=True
    )

    capacity_provided = evaluator.add_custom_node(
        result=bool(extracted.concert_capacity and extracted.concert_capacity.strip()),
        id="Concert_Capacity_Provided",
        desc="The concert capacity of the selected arena is provided.",
        parent=required_output_node,
        critical=True
    )

    city_provided = evaluator.add_custom_node(
        result=bool(extracted.city and extracted.city.strip()),
        id="City_Provided",
        desc="The city where the selected arena is located is specified.",
        parent=required_output_node,
        critical=True
    )

    # Node: Venue Selection Criteria (critical)
    selection_node = evaluator.add_parallel(
        id="Venue_Selection_Criteria",
        desc="The selected venue satisfies all selection criteria from the question/constraints.",
        parent=task_node,
        critical=True
    )

    # Prepare some values/sources
    arena_name = extracted.arena_full_name or ""
    # Merge source pools for robustness
    indoor_sources = _merge_sources(extracted.indoor_type_sources, extracted.general_sources, extracted.capacity_sources)
    distance_sources = _merge_sources(extracted.distance_sources, extracted.general_sources)
    highest_sources = _merge_sources(extracted.highest_capacity_sources, extracted.capacity_sources, extracted.general_sources)

    # Leaf: Indoor_Arena_Type (critical)
    indoor_leaf = evaluator.add_leaf(
        id="Indoor_Arena_Type",
        desc="The selected venue is an indoor arena (not an outdoor stadium or amphitheatre).",
        parent=selection_node,
        critical=True
    )
    indoor_claim = (
        f"The venue '{arena_name}' is an indoor arena. It is an enclosed indoor facility suitable for concerts "
        f"(not an open-air stadium or outdoor amphitheatre)."
    )
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_leaf,
        sources=indoor_sources if indoor_sources else None,
        additional_instruction=(
            "Use the provided URL(s) to confirm the venue type. Accept phrases like 'indoor arena', 'indoor multi-purpose arena', "
            "'enclosed', or clearly indicated roofed arena suitable for concerts. If the venue is a stadium or outdoor amphitheatre, "
            "or the sources do not support it being indoor, mark as not supported."
        ),
        extra_prerequisites=[name_provided]
    )

    # Leaf: Within_Five_Miles_of_Levis_Stadium (critical)
    distance_leaf = evaluator.add_leaf(
        id="Within_Five_Miles_of_Levis_Stadium",
        desc="The selected venue is located within 5 miles of Levi's Stadium in Santa Clara.",
        parent=selection_node,
        critical=True
    )
    distance_claim = (
        f"The distance between '{arena_name}' and {REFERENCE_STADIUM_NAME} in {REFERENCE_STADIUM_CITY} is within 5 miles (≤ 5.0 miles)."
    )
    await evaluator.verify(
        claim=distance_claim,
        node=distance_leaf,
        sources=distance_sources if distance_sources else None,
        additional_instruction=(
            "Prefer sources that explicitly show a distance (e.g., Google Maps driving/walking distance). "
            "Either straight-line or driving distance within 5.0 miles is acceptable for this check. "
            "If the provided webpage(s) do not display a distance ≤ 5 miles, consider the claim not supported."
        ),
        extra_prerequisites=[name_provided]
    )

    # Leaf: Highest_Concert_Capacity_Among_Qualifiers (critical)
    highest_leaf = evaluator.add_leaf(
        id="Highest_Concert_Capacity_Among_Qualifiers",
        desc="Among all indoor arenas within 5 miles of Levi's Stadium, the selected venue has the highest concert capacity.",
        parent=selection_node,
        critical=True
    )
    highest_claim = (
        f"Among indoor arenas located within 5 miles of {REFERENCE_STADIUM_NAME}, the venue '{arena_name}' "
        f"has the highest concert capacity."
    )
    await evaluator.verify(
        claim=highest_claim,
        node=highest_leaf,
        sources=highest_sources if highest_sources else None,
        additional_instruction=(
            "The supporting URLs should directly state or allow clear comparison that this venue has the highest concert capacity "
            "among indoor arenas within a 5-mile radius of Levi's Stadium. Evidence might include: (a) a list/table comparing "
            "capacities of relevant nearby indoor arenas; or (b) explicit statements about being the largest by capacity among "
            "indoor arenas in that radius. If the evidence is insufficient or only shows a single venue's capacity without "
            "comparison to other nearby indoor arenas, mark as not supported."
        ),
        extra_prerequisites=[name_provided, indoor_leaf, distance_leaf]
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
    Evaluate an answer for the indoor arena selection near Levi's Stadium task.

    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_arena_info(),
        template_class=ArenaExtraction,
        extraction_name="selected_arena_extraction"
    )

    # Build verification nodes and run checks
    await build_verification_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()