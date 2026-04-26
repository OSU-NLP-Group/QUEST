import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_esports_facility_na"
TASK_DESCRIPTION = """
What is the largest dedicated esports facility in North America? Provide the facility's name, its complete location (city, state, and street address), its seated spectator capacity, and its total size in square feet.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    """
    Structured extraction from the agent's answer for the largest dedicated esports facility in North America.
    """
    name: Optional[str] = None

    # Location details
    street_address: Optional[str] = None
    city: Optional[str] = None
    state_province: Optional[str] = None

    # Capacity and size (keep as strings to be robust to formatting like "about 2,500")
    seated_capacity: Optional[str] = None
    total_size_sqft: Optional[str] = None

    # Claim text (if explicitly stated in the answer)
    largest_claim_text: Optional[str] = None

    # Source URLs grouped by the aspect they support
    sources_largest: List[str] = Field(default_factory=list)
    sources_location: List[str] = Field(default_factory=list)
    sources_capacity: List[str] = Field(default_factory=list)
    sources_size: List[str] = Field(default_factory=list)

    # Any additional source URLs cited in the answer
    other_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_details() -> str:
    return """
    Extract the facility details as presented in the answer for the query:
    "What is the largest dedicated esports facility in North America? Provide the facility's name, its complete location (city, state, and street address), its seated spectator capacity, and its total size in square feet."

    Return a JSON object with the following fields:
    - name: The facility’s name (string). If not present, null.
    - street_address: The street address (string). If not present, null.
    - city: The city (string). If not present, null.
    - state_province: The U.S. state or Canadian province (string). If not present, null.
    - seated_capacity: The seated spectator capacity as it appears in the answer (string; may include formatting like "2,500"). If not present, null.
    - total_size_sqft: The total size in square feet as it appears in the answer (string; may include formatting like "100,000"). If not present, null.
    - largest_claim_text: The exact sentence or phrase in the answer that asserts the facility is the largest dedicated esports facility in North America. If not present, null.

    Also extract URLs cited in the answer, grouped by what they support:
    - sources_largest: URLs intended to support the "largest dedicated esports facility in North America" claim.
    - sources_location: URLs intended to support the full location details (street address, city, state/province).
    - sources_capacity: URLs intended to support the seated spectator capacity.
    - sources_size: URLs intended to support the total size in square feet.
    - other_sources: Any other URLs cited in the answer that do not neatly fit the above categories.

    SPECIAL RULES FOR URL EXTRACTION:
    - Only extract actual URLs explicitly present in the answer. They may be plain URLs or markdown links; if markdown links are present, extract their underlying URL.
    - If a URL is missing protocol (http/https), prepend http://.
    - Ignore obviously invalid or malformed URLs.

    If any of the requested fields are missing in the answer, return null for them; for URL arrays, return an empty array when nothing is cited.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _has_digits(s: Optional[str]) -> bool:
    return bool(s) and any(ch.isdigit() for ch in s)


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _combine_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for urls in url_lists:
        for u in urls:
            if isinstance(u, str):
                u_norm = u.strip()
                if u_norm and u_norm not in seen:
                    seen.add(u_norm)
                    combined.append(u_norm)
    return combined


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extraction: FacilityExtraction,
) -> None:
    """
    Build the verification tree from the rubric and conduct verifications.
    """
    # Create a critical parent node that mirrors the rubric's main node.
    main_node = evaluator.add_parallel(
        id="Largest_Dedicated_Esports_Facility_In_North_America",
        desc=(
            "Answer identifies the largest dedicated esports facility in North America and provides the required "
            "details (name, complete location, seated capacity, total size in sq ft) with authoritative verification."
        ),
        parent=evaluator.root,
        critical=True
    )

    # 1) Facility_Name_Provided (critical, existence check)
    evaluator.add_custom_node(
        result=_nonempty(extraction.name),
        id="Facility_Name_Provided",
        desc="Provides the facility’s name (a specific dedicated esports facility).",
        parent=main_node,
        critical=True
    )

    # 2) Complete_Location_Provided (critical, check that street, city, and state/province are present)
    complete_location_present = (
        _nonempty(extraction.street_address)
        and _nonempty(extraction.city)
        and _nonempty(extraction.state_province)
    )
    evaluator.add_custom_node(
        result=complete_location_present,
        id="Complete_Location_Provided",
        desc="Provides the complete location including city, state/province (as applicable), and street address.",
        parent=main_node,
        critical=True
    )

    # 3) Seated_Spectator_Capacity_Provided (critical, numeric-like check)
    evaluator.add_custom_node(
        result=_has_digits(extraction.seated_capacity),
        id="Seated_Spectator_Capacity_Provided",
        desc="Provides a seated spectator capacity (a numeric value).",
        parent=main_node,
        critical=True
    )

    # 4) Total_Size_SqFt_Provided (critical, numeric-like check)
    evaluator.add_custom_node(
        result=_has_digits(extraction.total_size_sqft),
        id="Total_Size_SqFt_Provided",
        desc="Provides the facility’s total size in square feet (a numeric value).",
        parent=main_node,
        critical=True
    )

    # 5) Authoritative_Sources_Cited (critical, verify presence and authority via cited URLs)
    auth_sources_node = evaluator.add_leaf(
        id="Authoritative_Sources_Cited",
        desc=(
            "Includes verifiable citations from official or otherwise authoritative sources supporting the key "
            "claims (largest designation, location, capacity, size)."
        ),
        parent=main_node,
        critical=True
    )

    all_sources = _combine_sources(
        extraction.sources_largest,
        extraction.sources_location,
        extraction.sources_capacity,
        extraction.sources_size,
        extraction.other_sources
    )

    if all_sources:
        auth_claim = (
            "Among the provided URLs, at least one page is an official or otherwise authoritative source "
            "(e.g., the facility's official website, a government/municipal page, or a major reputable publication), "
            "and it directly supports at least one key fact stated in the answer: "
            "the facility's 'largest dedicated' designation in North America, the exact street address and city/state, "
            "the seated spectator capacity, or the total size in square feet."
        )
        await evaluator.verify(
            claim=auth_claim,
            node=auth_sources_node,
            sources=all_sources,
            additional_instruction=(
                "Judge authority pragmatically: official site domains, city/government portals (.gov), "
                "recognized organizations, or reputable press with clear reporting are authoritative. "
                "Check whether the page explicitly supports any of the listed key facts from the answer."
            )
        )
    else:
        # No sources cited; this must fail.
        auth_sources_node.score = 0.0
        auth_sources_node.status = "failed"

    # 6) Largest_Dedicated_Claim_Supported (critical, verify claim via cited sources)
    largest_supported_node = evaluator.add_leaf(
        id="Largest_Dedicated_Claim_Supported",
        desc=(
            "Explicitly states the facility is the largest dedicated esports facility in North America and provides "
            "support (evidence/citation) for this claim."
        ),
        parent=main_node,
        critical=True
    )

    largest_claim_text = extraction.largest_claim_text or (
        f"The facility '{extraction.name}' is the largest dedicated esports facility in North America."
        if _nonempty(extraction.name) else
        "This facility is the largest dedicated esports facility in North America."
    )

    if extraction.sources_largest:
        await evaluator.verify(
            claim=largest_claim_text,
            node=largest_supported_node,
            sources=extraction.sources_largest,
            additional_instruction=(
                "Verify that the provided page explicitly supports 'largest dedicated esports facility in North America' "
                "for the named facility (allow minor phrasing variations like 'North America's largest esports facility'). "
                "Do not accept vague phrases like 'one of the largest' as sufficient."
            )
        )
    else:
        # No specific sources for the largest claim; fail this leaf.
        largest_supported_node.score = 0.0
        largest_supported_node.status = "failed"


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
    Evaluate an agent's answer for the largest dedicated esports facility in North America.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured facility details from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_facility_details(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction"
    )

    # Add a compact custom info entry for debugging/tracing
    evaluator.add_custom_info(
        info={
            "name": extraction.name,
            "street_address": extraction.street_address,
            "city": extraction.city,
            "state_province": extraction.state_province,
            "seated_capacity": extraction.seated_capacity,
            "total_size_sqft": extraction.total_size_sqft,
            "sources_largest_count": len(extraction.sources_largest),
            "sources_location_count": len(extraction.sources_location),
            "sources_capacity_count": len(extraction.sources_capacity),
            "sources_size_count": len(extraction.sources_size),
            "other_sources_count": len(extraction.other_sources),
        },
        info_type="extraction_summary",
        info_name="extracted_facility_summary"
    )

    # Build and verify the tree per rubric
    await build_and_verify_tree(evaluator, extraction)

    # Return standardized summary
    return evaluator.get_summary()