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
TASK_ID = "ca_coastal_rv_parks"
TASK_DESCRIPTION = (
    "Identify 3 California coastal state parks that offer RV camping with the following requirements: "
    "accommodate RVs up to at least 35 feet in length, provide RV sites with both water and electric hookups, "
    "have shower facilities available to campers, include a dump station for RV waste disposal, and are located "
    "directly on or adjacent to the California coast."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkItem(BaseModel):
    name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return (
        "Extract all California state parks mentioned in the answer that the answer claims meet the specified RV camping "
        "requirements. For each park, return:\n"
        "1. name: The exact park name as written in the answer.\n"
        "2. source_urls: A list of all URLs explicitly provided in the answer that are used as sources or evidence for this park. "
        "   Include URLs appearing as plain text or within markdown links. Only include valid URLs; if none are provided, return an empty list.\n"
        "Return a JSON object with a 'parks' array of objects {name, source_urls}. Extract in the order they appear in the answer. "
        "Do not invent or infer any URLs. If the answer lists more than 3 parks, still extract all; the evaluator will later limit to the first 3."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name.strip() if (name and name.strip()) else "the park"


def _additional_instruction_coastal(park_name: str) -> str:
    return (
        f"Confirm that {park_name} is a California State Park and that it is located on or directly adjacent to the "
        f"Pacific Ocean coastline in California (oceanfront/beachfront/coastal bluff or immediately next to the ocean). "
        f"Exclude inland parks (e.g., lakes or rivers not on the ocean). Use explicit evidence on the provided webpage(s). "
        f"If the page is irrelevant or does not substantiate coastal adjacency and state park status, mark as not supported."
    )


def _additional_instruction_rv_length(park_name: str) -> str:
    return (
        f"Confirm that campground(s) in {park_name} accommodate RVs up to at least 35 feet in overall length. "
        f"Look for phrases like 'maximum vehicle length', 'RV length', 'max trailer length', etc. "
        f"If multiple campgrounds have different limits, passing is acceptable if at least one accommodates ≥35 ft."
    )


def _additional_instruction_hookups(park_name: str) -> str:
    return (
        f"Confirm that {park_name} offers RV campsites with both water and electric hookups. "
        f"Accept 'full hookups' (which implies water + electric, often sewer too) or explicit mention of both water and electric. "
        f"If only electric or only water is available, or the page is unclear, mark as not supported."
    )


def _additional_instruction_showers(park_name: str) -> str:
    return (
        f"Confirm that showers are available to campers at {park_name}. "
        f"Accept coin-operated showers or any explicit mention of showers in the campground amenities."
    )


def _additional_instruction_dump(park_name: str) -> str:
    return (
        f"Confirm that an RV dump station (sanitation station) is available at {park_name} or within the park campground area."
    )


# --------------------------------------------------------------------------- #
# Park verification                                                           #
# --------------------------------------------------------------------------- #
async def verify_park(evaluator: Evaluator, parent_node, park: ParkItem, index: int) -> None:
    """
    Build verification nodes for a single park and run evidence-based checks.
    """
    park_name = _safe_name(park.name)
    park_node = evaluator.add_parallel(
        id=f"park_{index + 1}",
        desc=f"{['First','Second','Third'][index]} California coastal state park meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # Critical existence of park name (gates other checks)
    name_exists = bool(park.name and park.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"park_{index + 1}_name_provided",
        desc=f"Park #{index + 1} name is provided",
        parent=park_node,
        critical=True
    )

    # Optional: record whether sources were provided (non-critical for partial credit insight)
    evaluator.add_custom_node(
        result=bool(park.source_urls),
        id=f"park_{index + 1}_sources_provided",
        desc=f"Park #{index + 1} has at least one source URL provided",
        parent=park_node,
        critical=False
    )

    # Create leaf nodes for each requirement (all critical under the park node)
    coastal_leaf = evaluator.add_leaf(
        id=f"park_{index + 1}_coastal_location",
        desc="Park is a California state park located on or immediately adjacent to the California coast",
        parent=park_node,
        critical=True,
    )
    rv_length_leaf = evaluator.add_leaf(
        id=f"park_{index + 1}_rv_length",
        desc="Park accommodates RVs up to at least 35 feet in length",
        parent=park_node,
        critical=True,
    )
    hookups_leaf = evaluator.add_leaf(
        id=f"park_{index + 1}_hookups",
        desc="Park offers RV sites with both water and electric hookups",
        parent=park_node,
        critical=True,
    )
    showers_leaf = evaluator.add_leaf(
        id=f"park_{index + 1}_showers",
        desc="Park has shower facilities available to campers",
        parent=park_node,
        critical=True,
    )
    dump_leaf = evaluator.add_leaf(
        id=f"park_{index + 1}_dump_station",
        desc="Park includes a dump station for RV waste disposal",
        parent=park_node,
        critical=True,
    )

    claims_and_sources = [
        (
            f"{park_name} is a California State Park and is located on or directly adjacent to the California coast.",
            park.source_urls,
            coastal_leaf,
            _additional_instruction_coastal(park_name),
        ),
        (
            f"Campgrounds in {park_name} accommodate RVs up to at least 35 feet in length.",
            park.source_urls,
            rv_length_leaf,
            _additional_instruction_rv_length(park_name),
        ),
        (
            f"{park_name} offers RV campsites with both water and electric hookups.",
            park.source_urls,
            hookups_leaf,
            _additional_instruction_hookups(park_name),
        ),
        (
            f"Showers are available to campers at {park_name}.",
            park.source_urls,
            showers_leaf,
            _additional_instruction_showers(park_name),
        ),
        (
            f"{park_name} has an RV dump station available.",
            park.source_urls,
            dump_leaf,
            _additional_instruction_dump(park_name),
        ),
    ]

    # Verify all requirement claims (parallel within the park)
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the answer for California coastal RV parks with specific requirements.
    """
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

    # Record requirement summary for transparency
    evaluator.add_ground_truth({
        "requirements": [
            "California State Park on or adjacent to the ocean coast",
            "Accommodates RVs >= 35 ft",
            "RV sites include both water and electric hookups",
            "Showers available",
            "Dump station available",
        ],
        "count_required": 3
    }, gt_type="task_requirements")

    # Extract parks mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # Limit to first 3 parks; pad with empty entries if fewer
    parks = list(extracted.parks[:3])
    while len(parks) < 3:
        parks.append(ParkItem())

    # Build verification nodes for each park
    for i in range(3):
        await verify_park(evaluator, root, parks[i], i)

    return evaluator.get_summary()