import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "venue_identification_2025_sbs_gayo"
TASK_DESCRIPTION = (
    "What is the name of the venue that meets all of the following criteria as of December 31, 2025: "
    "Located in Incheon, South Korea; Has a seating capacity of 15,000; Hosted the 2025 SBS Gayo Daejeon music festival "
    "on December 25, 2025; Is part of an entertainment resort; Is Korea's first dedicated multi-purpose performance venue; "
    "The December 25, 2025 event was hosted by three MCs from the K-pop groups NCT, TXT, and IVE; Is located on Yeongjong Island. "
    "Provide the official name of this venue."
)


# --------------------------------------------------------------------------- #
# Data Models for Extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured information about the identified venue extracted from the answer."""
    venue_name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    island: Optional[str] = None
    capacity: Optional[str] = None
    resort_name: Optional[str] = None
    event_name: Optional[str] = None
    event_date: Optional[str] = None
    mc_groups: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the venue information presented in the answer. Return a single JSON object with these fields:
    - venue_name: The official name of the venue (as written in the answer; do not invent).
    - city: The city where the venue is located (if mentioned).
    - country: The country where the venue is located (if mentioned).
    - island: The island name if specified (e.g., Yeongjong or Yeongjong-do), otherwise null.
    - capacity: The stated seating capacity text for the venue (e.g., "15,000", "15000", "15K"); return as a string exactly as in the answer.
    - resort_name: The name of the entertainment or integrated resort the venue belongs to (if present), otherwise null.
    - event_name: The event name related to December 25, 2025 (e.g., "SBS Gayo Daejeon") if provided.
    - event_date: The date of the event as stated in the answer (prefer "December 25, 2025" or ISO-like), otherwise null.
    - mc_groups: The K-pop group names that the answer claims the MCs are from for the December 25, 2025 event. Only include group names (e.g., "NCT", "TXT", "IVE"); do not include individual MC names. If unclear, return an empty list.
    - sources: A list of all URLs explicitly cited in the answer as evidence for any of the above claims. Include every URL mentioned in the answer (plain or markdown).
    If any field is missing from the answer, set it to null (or an empty array for lists). Do not add information not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification Logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_venue_criteria(evaluator: Evaluator, parent_node, info: VenueExtraction) -> None:
    """
    Build the verification tree and run checks for each rubric criterion.
    All factual checks are grounded by the answer's cited URLs whenever possible.
    """
    # Create the main node corresponding to the rubric's top-level item
    venue_node = evaluator.add_parallel(
        id="Venue_Identification",
        desc="Correctly identify the venue that meets all specified criteria",
        parent=parent_node,
        critical=False  # Non-critical root for partial scoring across sub-criteria
    )

    # Gate: ensure we have a venue name and at least one source URL
    has_name = isinstance(info.venue_name, str) and info.venue_name.strip() != ""
    has_sources = isinstance(info.sources, list) and len([u for u in info.sources if isinstance(u, str) and u.strip()]) > 0
    evaluator.add_custom_node(
        result=has_name and has_sources,
        id="venue_name_and_sources_provided",
        desc="Venue name and at least one source URL are provided in the answer",
        parent=venue_node,
        critical=True
    )

    name = info.venue_name or ""

    # 1) Location in Incheon, South Korea
    node_loc_incheon = evaluator.add_leaf(
        id="Location_Incheon",
        desc="The venue is located in Incheon, South Korea",
        parent=venue_node,
        critical=True
    )
    claim_loc_incheon = f"The venue named '{name}' is located in Incheon, South Korea."
    await evaluator.verify(
        claim=claim_loc_incheon,
        node=node_loc_incheon,
        sources=info.sources,
        additional_instruction=(
            "Confirm that the venue's location is within Incheon (Incheon Metropolitan City), South Korea. "
            "Phrases like 'Incheon, South Korea' or 'Incheon, Republic of Korea' both qualify. "
            "Mentions like 'Yeongjong-do, Incheon' should also be considered as Incheon."
        )
    )

    # 2) Seating capacity of 15,000
    node_capacity = evaluator.add_leaf(
        id="Capacity_15000",
        desc="The venue has a seating capacity of 15,000",
        parent=venue_node,
        critical=True
    )
    claim_capacity = f"The venue named '{name}' has a seating capacity of 15,000."
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity,
        sources=info.sources,
        additional_instruction=(
            "Accept reasonable equivalents such as '15,000 seats', 'capacity 15,000', '15k', or 'approximately 15,000'. "
            "The claim should clearly indicate the venue's capacity is 15,000 (seats)."
        )
    )

    # 3) Hosted the 2025 SBS Gayo Daejeon on December 25, 2025
    node_event = evaluator.add_leaf(
        id="Event_SBS_Gayo_Daejeon",
        desc="The venue hosted the 2025 SBS Gayo Daejeon on December 25, 2025",
        parent=venue_node,
        critical=True
    )
    claim_event = (
        f"On December 25, 2025, the 2025 SBS Gayo Daejeon music festival was held at the venue named '{name}'."
    )
    await evaluator.verify(
        claim=claim_event,
        node=node_event,
        sources=info.sources,
        additional_instruction=(
            "Verify both the event name (SBS Gayo Daejeon 2025) and the date (December 25, 2025) are associated with this venue. "
            "Support can come from official announcements, reputable news articles, or the venue/resort's site."
        )
    )

    # 4) Part of an entertainment resort
    node_resort = evaluator.add_leaf(
        id="Resort_Location",
        desc="The venue is part of an entertainment resort",
        parent=venue_node,
        critical=True
    )
    if info.resort_name and info.resort_name.strip():
        claim_resort = f"The venue named '{name}' is part of the entertainment (integrated) resort '{info.resort_name.strip()}'."
    else:
        claim_resort = f"The venue named '{name}' is part of an entertainment or integrated resort complex."
    await evaluator.verify(
        claim=claim_resort,
        node=node_resort,
        sources=info.sources,
        additional_instruction=(
            "Accept synonyms like 'entertainment resort', 'integrated resort', or 'resort complex'. "
            "Ideally, the source explicitly links the venue to the named resort."
        )
    )

    # 5) Korea's first dedicated multi-purpose performance venue
    node_first = evaluator.add_leaf(
        id="First_Multipurpose_Venue",
        desc="The venue is Korea's first dedicated multi-purpose performance venue",
        parent=venue_node,
        critical=True
    )
    claim_first = (
        f"The venue named '{name}' is South Korea's first dedicated multi-purpose performance venue (or performance arena)."
    )
    await evaluator.verify(
        claim=claim_first,
        node=node_first,
        sources=info.sources,
        additional_instruction=(
            "Look for phrasings such as 'Korea's first multi-purpose performance venue', "
            "'the nation's first dedicated performance arena', or closely equivalent wording indicating first-of-its-kind status."
        )
    )

    # 6) Event hosted by three MCs from NCT, TXT, and IVE
    node_three_mcs = evaluator.add_leaf(
        id="Three_MCs",
        desc="The December 25, 2025 event was hosted by three MCs from NCT, TXT, and IVE respectively",
        parent=venue_node,
        critical=True
    )
    claim_three_mcs = (
        "The December 25, 2025 SBS Gayo Daejeon was hosted by three MCs, with one MC from each K-pop group: NCT, TXT, and IVE."
    )
    await evaluator.verify(
        claim=claim_three_mcs,
        node=node_three_mcs,
        sources=info.sources,
        additional_instruction=(
            "Confirm that the event lists three hosts/MCs, and that their group affiliations include one member from NCT, one from TXT, and one from IVE. "
            "Exact individual names are not required to be verified here as long as the group affiliations are correct."
        )
    )

    # 7) Located on Yeongjong Island
    node_island = evaluator.add_leaf(
        id="Island_Location",
        desc="The venue is located on Yeongjong Island",
        parent=venue_node,
        critical=True
    )
    claim_island = f"The venue named '{name}' is located on Yeongjong Island (also known as Yeongjong-do) in Incheon."
    await evaluator.verify(
        claim=claim_island,
        node=node_island,
        sources=info.sources,
        additional_instruction=(
            "Accept mentions of 'Yeongjong', 'Yeongjong-do', or 'Yeongjong Island' as equivalent. "
            "Ensure the island reference explicitly pertains to the venue's location."
        )
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an answer for identifying the venue meeting all specified criteria (SBS Gayo Daejeon 2025 venue).
    """
    # Initialize evaluator and root
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

    # Extract structured venue info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Optional: record helpful context
    evaluator.add_custom_info(
        info={
            "extracted_venue_name": extracted.venue_name,
            "resort_name": extracted.resort_name,
            "source_count": len(extracted.sources) if extracted.sources else 0
        },
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    # Build verification tree and run checks
    await verify_venue_criteria(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()