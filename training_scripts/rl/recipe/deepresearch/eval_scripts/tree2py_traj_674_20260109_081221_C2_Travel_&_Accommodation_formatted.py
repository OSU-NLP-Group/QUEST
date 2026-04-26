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
TASK_ID = "atl_minute_suites_overnight_shower"
TASK_DESCRIPTION = (
    "At Hartsfield-Jackson Atlanta International Airport, which specific Minute Suites location "
    "(concourse and gate number) offers shower facilities and supports overnight bookings (8-hour stays after 9 PM)?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MinuteSuitesATLExtraction(BaseModel):
    """
    Structured extraction for the specific Minute Suites location at ATL and its features.
    """
    airport: Optional[str] = None
    concourse: Optional[str] = None
    gate_number: Optional[str] = None
    location_label: Optional[str] = None  # e.g., "Concourse B near Gate B24" as stated in the answer
    location_sources: List[str] = Field(default_factory=list)  # URLs cited for location identification
    shower_sources: List[str] = Field(default_factory=list)    # URLs cited for shower availability
    overnight_sources: List[str] = Field(default_factory=list) # URLs cited for overnight booking capability


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_minute_suites_atl() -> str:
    return """
    From the answer, extract the specific Minute Suites location at Hartsfield-Jackson Atlanta International Airport (ATL) that is claimed to offer shower facilities and to support overnight bookings (defined here as 8-hour stays after 9 PM).

    Required fields:
    - airport: The airport code or name mentioned (e.g., "ATL" or "Hartsfield-Jackson Atlanta International Airport"). If not explicitly stated, return null.
    - concourse: The concourse letter or identifier for the Minute Suites location (e.g., "Concourse B"). If not explicitly stated, return null.
    - gate_number: The gate number or gate label as shown in the answer (e.g., "B24" or "Gate B24"). Keep the exact format used in the answer. If not explicitly stated, return null.
    - location_label: If the answer provides a combined label (e.g., "Concourse B near Gate B24"), extract it; otherwise, return null.

    Also extract the following URL lists, strictly as they appear in the answer:
    - location_sources: All URL(s) cited that support the specific concourse and gate identification of the Minute Suites location at ATL.
    - shower_sources: All URL(s) cited that specifically support shower availability at that location.
    - overnight_sources: All URL(s) cited that specifically support overnight bookings (8-hour stays after 9 PM) at that location.

    Rules for sources:
    - Extract only actual URLs explicitly present in the answer (plain URLs or markdown links).
    - Do not invent URLs. If a source is referenced without a URL, do not include it.
    - If a URL is missing a protocol, prepend http://.
    - If no URLs are provided for a category, return an empty list for that category.

    If any required field (airport, concourse, gate_number) is missing in the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists preserving order and uniqueness."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _loc_str(concourse: Optional[str], gate_number: Optional[str], fallback_label: Optional[str]) -> str:
    """Build a human-readable location string for claims."""
    if concourse and gate_number:
        return f"Concourse {concourse.replace('Concourse ', '').strip()} near Gate {gate_number.strip()}"
    if fallback_label:
        return fallback_label.strip()
    # Minimal placeholder text to allow a claim if fields are missing
    return "the specified Minute Suites location at ATL"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_location(
    evaluator: Evaluator,
    parent_node,
    extracted: MinuteSuitesATLExtraction
) -> None:
    """
    Build the 'location_identification' subtree and run checks.
    """
    # Parent node: location identification (critical)
    loc_node = evaluator.add_parallel(
        id="location_identification",
        desc="Provide the specific Minute Suites location at Hartsfield-Jackson Atlanta International Airport (ATL) by concourse and gate number.",
        parent=parent_node,
        critical=True
    )

    # 1) Existence check: Both concourse and gate_number must be present
    fields_present = bool(extracted.concourse and extracted.concourse.strip()) and bool(extracted.gate_number and extracted.gate_number.strip())
    evaluator.add_custom_node(
        result=fields_present,
        id="location_fields_present",
        desc="Concourse and gate number are provided in the answer.",
        parent=loc_node,
        critical=True
    )

    # 2) Source-backed verification that the identified location is correct
    loc_leaf = evaluator.add_leaf(
        id="location_supported_by_sources",
        desc="The identified Minute Suites location at ATL (concourse + gate) is supported by cited sources.",
        parent=loc_node,
        critical=True
    )

    location_string = _loc_str(extracted.concourse, extracted.gate_number, extracted.location_label)
    loc_claim = f"The Minute Suites location at Hartsfield-Jackson Atlanta International Airport (ATL) is at {location_string}."
    location_urls = extracted.location_sources
    # fall back to feature sources if no dedicated location sources are provided
    if not location_urls:
        location_urls = _merge_sources(extracted.shower_sources, extracted.overnight_sources)

    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=location_urls,
        additional_instruction=(
            "Verify that the provided sources explicitly state the Minute Suites ATL location's concourse and gate. "
            "Allow minor formatting differences (e.g., 'Gate B24' vs 'B24'). If multiple ATL locations exist, "
            "ensure the sources correspond to the one specified."
        )
    )


async def build_and_verify_features(
    evaluator: Evaluator,
    parent_node,
    extracted: MinuteSuitesATLExtraction
) -> None:
    """
    Build the 'feature_verification' subtree and run checks for showers and overnight capability.
    """
    feat_node = evaluator.add_parallel(
        id="feature_verification",
        desc="Verify the identified location satisfies both required features.",
        parent=parent_node,
        critical=True
    )

    # Shower availability
    shower_leaf = evaluator.add_leaf(
        id="shower_availability",
        desc="Confirm the identified location offers shower facilities.",
        parent=feat_node,
        critical=True
    )
    location_string = _loc_str(extracted.concourse, extracted.gate_number, extracted.location_label)
    shower_claim = f"The Minute Suites location at ATL ({location_string}) offers shower facilities."
    shower_urls = extracted.shower_sources or _merge_sources(extracted.location_sources)
    await evaluator.verify(
        claim=shower_claim,
        node=shower_leaf,
        sources=shower_urls,
        additional_instruction=(
            "Confirm that the cited page(s) explicitly indicate showers are available for the same ATL Minute Suites location "
            "specified (matching concourse/gate or a clearly tied ATL unit). General brand pages without location-specific "
            "evidence should not be considered sufficient."
        )
    )

    # Overnight capability (8-hour stays after 9 PM)
    overnight_leaf = evaluator.add_leaf(
        id="overnight_capability",
        desc="Confirm the identified location supports overnight bookings (8-hour stays after 9 PM).",
        parent=feat_node,
        critical=True
    )
    overnight_claim = (
        f"The Minute Suites location at ATL ({location_string}) supports overnight bookings defined as 8-hour stays starting after 9 PM."
    )
    overnight_urls = extracted.overnight_sources or _merge_sources(extracted.location_sources)
    await evaluator.verify(
        claim=overnight_claim,
        node=overnight_leaf,
        sources=overnight_urls,
        additional_instruction=(
            "Look for explicit mentions of 'Overnight' or '8-hour after 9 PM' booking options (or equivalent phrases) "
            "for the specific ATL location. Policy pages that clearly apply to this specific unit are acceptable."
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
    Evaluate the agent's answer for the ATL Minute Suites location with shower and overnight features.
    """
    # Initialize evaluator with sequential root to enforce dependency
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_minute_suites_atl(),
        template_class=MinuteSuitesATLExtraction,
        extraction_name="minute_suites_atl_extraction"
    )

    # Build verification tree according to rubric
    # 1) Location identification (critical)
    await build_and_verify_location(evaluator, root, extracted)

    # 2) Feature verification (critical, parallel children)
    await build_and_verify_features(evaluator, root, extracted)

    # Optionally record custom info for debugging
    evaluator.add_custom_info(
        {
            "airport": extracted.airport,
            "concourse": extracted.concourse,
            "gate_number": extracted.gate_number,
            "location_label": extracted.location_label,
            "location_sources": extracted.location_sources,
            "shower_sources": extracted.shower_sources,
            "overnight_sources": extracted.overnight_sources
        },
        info_type="extraction_debug",
        info_name="extracted_fields"
    )

    # Return the structured evaluation summary
    return evaluator.get_summary()