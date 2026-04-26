import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_esports_venue_na"
TASK_DESCRIPTION = """
Identify the largest dedicated esports venue in North America by square footage. Provide the venue's name, seating capacity, total square footage, city and state location, and the year it opened.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured info for the chosen largest dedicated esports venue."""
    venue_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    total_square_footage: Optional[str] = None
    city: Optional[str] = None
    state_or_province: Optional[str] = None
    country: Optional[str] = None
    opening_year: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    From the provided answer, extract the details of the SINGLE venue that the answer claims is
    the largest dedicated esports venue in North America (by total square footage).

    Return the following fields:
    - venue_name: The venue's official name as written in the answer.
    - seating_capacity: The spectator seating capacity as written (string; keep formatting and units if present).
    - total_square_footage: The total venue size in square feet as written (string; keep formatting/units if present).
    - city: The city where the venue is located, if stated.
    - state_or_province: The state or province where the venue is located, if stated.
    - country: The country where the venue is located, if stated.
    - opening_year: The year the venue opened/inaugurated (4-digit year, as written).
    - source_urls: All URLs explicitly cited in the answer that support this venue and its claims.
      These can be plain URLs or markdown links; extract the actual URLs.

    RULES:
    - Do not invent information. If a field is not present in the answer, return null (or [] for lists).
    - If multiple venues are mentioned, extract only the one the answer treats as the "largest dedicated esports venue in North America".
    - Preserve strings exactly as presented in the answer; do not normalize units or numbers.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_digits(s: Optional[str]) -> bool:
    return bool(s and any(ch.isdigit() for ch in s))


def _has_four_digit_year(s: Optional[str]) -> bool:
    return bool(s and re.search(r"\b(19|20)\d{2}\b", s))


def _compose_location(city: Optional[str], state_or_province: Optional[str], country: Optional[str]) -> str:
    parts = [p for p in [city, state_or_province, country] if p and str(p).strip()]
    return ", ".join(parts) if parts else "Unknown location"


def _normalize_sources_for_verify(sources: List[str]) -> List[str] | None:
    """Return list if non-empty, else None (causes simple verification)."""
    return sources if sources else None


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: VenueExtraction,
) -> None:
    """
    Build the rubric tree and run verifications according to the provided rubric.
    All nodes under the rubric root are critical, meaning failure will fail the parent.
    """

    # Root rubric node (critical, parallel aggregation)
    rubric_root = evaluator.add_parallel(
        id="Largest_Dedicated_Esports_Venue_Identification",
        desc="Identify the largest dedicated esports venue in North America by square footage and provide the required venue details.",
        parent=evaluator.root,
        critical=True
    )

    # Group: Venue qualifies and is largest (critical, parallel)
    qualifies_node = evaluator.add_parallel(
        id="Venue_Qualifies_And_Is_Largest",
        desc="The chosen venue satisfies the definition of a dedicated esports facility in North America and is the largest by total square footage among such venues.",
        parent=rubric_root,
        critical=True
    )

    # Sources to use for verification (may be empty; framework will fallback to simple verification)
    verify_sources = _normalize_sources_for_verify(extracted.source_urls)

    # 1) Located in North America
    located_leaf = evaluator.add_leaf(
        id="Located_In_North_America",
        desc="Venue is located in North America.",
        parent=qualifies_node,
        critical=True
    )
    venue_name_for_claim = extracted.venue_name or "the venue"
    location_desc = _compose_location(extracted.city, extracted.state_or_province, extracted.country)
    located_claim = (
        f"{venue_name_for_claim} is located in North America (United States, Canada, or Mexico). "
        f"The location mentioned/indicated is: {location_desc}."
    )
    await evaluator.verify(
        claim=located_claim,
        node=located_leaf,
        sources=verify_sources,
        additional_instruction=(
            "Confirm from the webpage content that the venue is in a city within the United States, Canada, or Mexico. "
            "Allow reasonable location phrasing variants. If the webpage clearly shows a US/Canada/Mexico location, mark as supported."
        )
    )

    # 2) Dedicated esports facility
    dedicated_leaf = evaluator.add_leaf(
        id="Dedicated_Esports_Facility",
        desc="Venue is a dedicated esports facility (not merely a general-purpose venue hosting esports).",
        parent=qualifies_node,
        critical=True
    )
    dedicated_claim = (
        f"{venue_name_for_claim} is a dedicated esports facility, primarily built for or operated for esports competitions "
        f"and events, rather than a general-purpose stadium that occasionally hosts esports."
    )
    await evaluator.verify(
        claim=dedicated_claim,
        node=dedicated_leaf,
        sources=verify_sources,
        additional_instruction=(
            "Look for explicit language that the venue is an 'esports stadium', 'esports arena', "
            "'dedicated esports facility', or similar. If it is clearly multi-purpose (e.g., a general arena that sometimes hosts esports), mark as not supported."
        )
    )

    # 3) Purpose-built or specifically designed for esports
    purpose_built_leaf = evaluator.add_leaf(
        id="Purpose_Built_Or_Designed_For_Esports",
        desc="Venue is purpose-built or specifically designed for esports competitions (not a general arena temporarily converted).",
        parent=qualifies_node,
        critical=True
    )
    purpose_built_claim = (
        f"{venue_name_for_claim} was purpose-built or specifically designed for esports competitions."
    )
    await evaluator.verify(
        claim=purpose_built_claim,
        node=purpose_built_leaf,
        sources=verify_sources,
        additional_instruction=(
            "The webpage should describe that the venue was built or designed specifically for esports (e.g., born as an esports arena/stadium). "
            "Phrases like 'purpose-built', 'designed for esports', or equivalent should appear or be clearly implied."
        )
    )

    # 4) Largest by square footage in North America among dedicated esports venues
    largest_leaf = evaluator.add_leaf(
        id="Largest_By_Square_Footage",
        desc="Venue is factually the largest dedicated esports venue in North America when comparing total square footage.",
        parent=qualifies_node,
        critical=True
    )
    sqft_text = extracted.total_square_footage or "the stated square footage"
    largest_claim = (
        f"{venue_name_for_claim} is the largest dedicated esports venue in North America by total square footage "
        f"(area: {sqft_text})."
    )
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        sources=verify_sources,
        additional_instruction=(
            "Check whether the page explicitly claims that this is the largest dedicated esports venue in North America by size/square footage "
            "or an equivalent assertion (e.g., 'largest in North America' combined with square footage context). "
            "If such a clear claim cannot be found or is contradicted, mark as not supported."
        )
    )

    # Field presence checks (critical, simple existence)
    name_provided = evaluator.add_custom_node(
        result=bool(extracted.venue_name and extracted.venue_name.strip()),
        id="Venue_Name_Provided",
        desc="Venue name is provided.",
        parent=rubric_root,
        critical=True
    )

    capacity_provided = evaluator.add_custom_node(
        result=_has_digits(extracted.seating_capacity),
        id="Seating_Capacity_Provided",
        desc="Seating capacity is provided as a measurable number of spectator seats.",
        parent=rubric_root,
        critical=True
    )

    sqft_provided = evaluator.add_custom_node(
        result=_has_digits(extracted.total_square_footage),
        id="Total_Square_Footage_Provided",
        desc="Total venue size is provided as a measurable value in square feet.",
        parent=rubric_root,
        critical=True
    )

    city_state_provided = evaluator.add_custom_node(
        result=bool(extracted.city and extracted.city.strip()) and bool(extracted.state_or_province and extracted.state_or_province.strip()),
        id="City_And_State_Provided",
        desc="City and state location are provided.",
        parent=rubric_root,
        critical=True
    )

    opening_year_provided = evaluator.add_custom_node(
        result=_has_four_digit_year(extracted.opening_year),
        id="Opening_Year_Provided",
        desc="Year the venue opened/was inaugurated is provided as a specific year.",
        parent=rubric_root,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    Evaluate an answer for the largest dedicated esports venue in North America task.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel; our rubric main node is added under this root
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
    extracted_venue = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="largest_esports_venue_info"
    )

    # Optionally record custom info snapshot to help debugging
    evaluator.add_custom_info(
        info={
            "extracted_name": extracted_venue.venue_name,
            "extracted_sqft": extracted_venue.total_square_footage,
            "extracted_location": _compose_location(
                extracted_venue.city, extracted_venue.state_or_province, extracted_venue.country
            ),
            "opening_year": extracted_venue.opening_year,
            "source_url_count": len(extracted_venue.source_urls or []),
        },
        info_type="extraction_summary",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted_venue)

    # Return evaluation summary
    return evaluator.get_summary()