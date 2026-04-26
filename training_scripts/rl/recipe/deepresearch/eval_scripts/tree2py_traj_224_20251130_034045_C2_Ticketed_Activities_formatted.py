import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "matt_rife_dec2025_florida_venue"
TASK_DESCRIPTION = (
    "Comedian Matt Rife is performing his Stay Golden Tour at a major arena in Florida during December 2025. "
    "Identify the name of this venue and provide its seating capacity. The venue's capacity must be at least 18,000 seats. "
    "Include reference URLs for both the tour date information and the venue capacity."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueCapacityExtraction(BaseModel):
    """
    Extracted information from the agent's answer for Matt Rife's December 2025 Florida show
    and the venue capacity with URLs.
    """
    venue_name: Optional[str] = None
    venue_location_text: Optional[str] = None  # City/State snippet as stated in answer (e.g., "Tampa, FL")
    performance_date_text: Optional[str] = None  # As stated, e.g., "December 12, 2025"
    stay_golden_tour_label: Optional[str] = None  # As stated, e.g., "Stay Golden Tour"; null if not mentioned

    tour_date_urls: List[str] = Field(default_factory=list)  # URLs for date/venue info (venue or official ticketing)
    capacity_number_text: Optional[str] = None  # Capacity number in text (e.g., "20,000")
    capacity_urls: List[str] = Field(default_factory=list)  # URLs supporting capacity (official or authoritative)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_capacity() -> str:
    return """
    Extract the following fields exactly as they appear in the answer:

    1) venue_name: The venue name hosting Matt Rife in Florida in December 2025.
    2) venue_location_text: Any location string provided for the venue (e.g., city/state such as "Tampa, FL").
    3) performance_date_text: The performance date string for the Florida show in December 2025 (e.g., "December 12, 2025").
    4) stay_golden_tour_label: The exact text in the answer that labels the tour as "Stay Golden Tour". If the answer does not explicitly mention this label, return null.
    5) tour_date_urls: All reference URLs provided in the answer that support the tour date/venue information. These should be official ticketing or venue sources when available.
    6) capacity_number_text: The venue seating capacity as provided in the answer (keep the text exactly, including commas or formatting).
    7) capacity_urls: All reference URLs provided in the answer that support the stated seating capacity.

    Rules:
    - Return null for any missing field.
    - For URL fields, extract only actual URLs present in the answer (plain or markdown). Do not invent or infer URLs.
    - Include all URLs relevant to the specific field; do not mix tour date URLs with capacity URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_urls(urls: List[str]) -> List[str]:
    """Trim and deduplicate URLs."""
    cleaned = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def parse_capacity_to_int(text: Optional[str]) -> Optional[int]:
    """
    Parse an integer capacity from free-text like "18,000", "20,500 seats", or "approx. 19,000".
    Returns the largest integer found if multiple appear.
    """
    if not text:
        return None
    candidates = re.findall(r"\d{1,3}(?:,\d{3})+|\d+", text)
    if not candidates:
        return None
    numbers = []
    for c in candidates:
        try:
            numbers.append(int(c.replace(",", "")))
        except Exception:
            continue
    return max(numbers) if numbers else None


def union_urls(*url_lists: List[str]) -> List[str]:
    """Union multiple URL lists after normalization."""
    combined: List[str] = []
    for lst in url_lists:
        combined.extend(lst)
    return normalize_urls(combined)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: VenueCapacityExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and perform checks.
    """

    # Root-level aggregation: Keep sequential to gate capacity verification after venue/show identification
    main_seq = evaluator.add_sequential(
        id="Venue_and_Capacity_Identification",
        desc="Identify the Florida venue hosting Matt Rife's Stay Golden Tour in December 2025 and provide a seating capacity (>=18,000) with supporting URLs.",
        parent=evaluator.root,
        critical=True,
    )

    # ------------------- Venue & Show Identification (Parallel) -------------------
    venue_show_node = evaluator.add_parallel(
        id="Venue_And_Show_Identification",
        desc="Identify the qualifying December 2025 Florida show/venue for Matt Rife's Stay Golden Tour and cite an official source for the date/venue.",
        parent=main_seq,
        critical=True,
    )

    # Venue name provided (existence)
    venue_name_provided = evaluator.add_custom_node(
        result=bool(extracted.venue_name and extracted.venue_name.strip()),
        id="Venue_Name_Provided",
        desc="The answer provides the venue name.",
        parent=venue_show_node,
        critical=True,
    )

    # Tour date reference URL provided (existence)
    tour_date_urls_clean = normalize_urls(extracted.tour_date_urls)
    tour_date_ref_provided = evaluator.add_custom_node(
        result=bool(tour_date_urls_clean),
        id="Tour_Date_Reference_URL_Provided",
        desc="A valid reference URL from an official ticketing or venue source is provided to verify the tour date/venue information.",
        parent=venue_show_node,
        critical=True,
    )

    # Located in Florida (verify via URLs)
    located_in_florida_leaf = evaluator.add_leaf(
        id="Located_In_Florida",
        desc="The identified venue is located in Florida.",
        parent=venue_show_node,
        critical=True,
    )
    venue_name_for_claim = extracted.venue_name or "the venue"
    await evaluator.verify(
        claim=f"The venue '{venue_name_for_claim}' is located in Florida.",
        node=located_in_florida_leaf,
        sources=union_urls(tour_date_urls_clean, normalize_urls(extracted.capacity_urls)),
        additional_instruction=(
            "Confirm the venue page or event listing shows a Florida location (city in Florida or 'FL'). "
            "Minor naming variants are acceptable."
        ),
    )

    # Performance occurs in December 2025 (verify via tour date URLs)
    performance_in_dec_leaf = evaluator.add_leaf(
        id="Performance_In_December_2025",
        desc="The cited performance date occurs in December 2025.",
        parent=venue_show_node,
        critical=True,
    )
    perf_date_text = extracted.performance_date_text or "the performance date"
    await evaluator.verify(
        claim=f"The referenced sources show a Matt Rife performance date in December 2025 (e.g., '{perf_date_text}').",
        node=performance_in_dec_leaf,
        sources=tour_date_urls_clean,
        additional_instruction=(
            "Check the event date on the page. Accept clearly indicated December 2025 dates "
            "(e.g., 'Dec 12, 2025', 'December 2025')."
        ),
    )

    # Part of Stay Golden Tour (verify via tour date URLs)
    stay_golden_leaf = evaluator.add_leaf(
        id="Part_Of_Stay_Golden_Tour",
        desc="The performance is identified as part of Matt Rife's Stay Golden Tour.",
        parent=venue_show_node,
        critical=True,
    )
    tour_label_text = extracted.stay_golden_tour_label or "Stay Golden Tour"
    await evaluator.verify(
        claim=f"The event is part of Matt Rife's '{tour_label_text}'.",
        node=stay_golden_leaf,
        sources=tour_date_urls_clean,
        additional_instruction=(
            "Look for explicit mention of 'Stay Golden Tour' or substantially equivalent phrasing "
            "on the event/venue/ticket page."
        ),
    )

    # ------------------- Capacity Verification (Parallel) -------------------
    capacity_node = evaluator.add_parallel(
        id="Capacity_Verification",
        desc="Provide and verify the seating capacity for the same identified venue, meeting the >=18,000 requirement, with a supporting URL.",
        parent=main_seq,
        critical=True,
    )

    # Capacity number provided (existence as a specific number)
    capacity_int = parse_capacity_to_int(extracted.capacity_number_text)
    capacity_number_provided = evaluator.add_custom_node(
        result=capacity_int is not None,
        id="Capacity_Number_Provided",
        desc="The answer provides the venue seating capacity as a specific number.",
        parent=capacity_node,
        critical=True,
    )

    # Capacity at least 18,000
    capacity_ge_18000 = evaluator.add_custom_node(
        result=(capacity_int is not None and capacity_int >= 18000),
        id="Capacity_At_Least_18000",
        desc="The provided seating capacity is at least 18,000 seats.",
        parent=capacity_node,
        critical=True,
    )

    # Capacity reference URL provided (existence)
    capacity_urls_clean = normalize_urls(extracted.capacity_urls)
    capacity_ref_provided = evaluator.add_custom_node(
        result=bool(capacity_urls_clean),
        id="Capacity_Reference_URL_Provided",
        desc="A valid reference URL is provided that supports the stated seating capacity (e.g., official venue page or equivalent authoritative source).",
        parent=capacity_node,
        critical=True,
    )

    # Capacity matches identified venue (verify via capacity URLs)
    capacity_matches_leaf = evaluator.add_leaf(
        id="Capacity_Matches_Identified_Venue",
        desc="The cited capacity corresponds to the same venue identified in the venue/show identification step.",
        parent=capacity_node,
        critical=True,
    )
    capacity_claim = (
        f"The seating capacity of '{venue_name_for_claim}' is approximately {capacity_int} seats "
        f"as shown on the provided capacity source(s)."
        if capacity_int is not None
        else f"The seating capacity of '{venue_name_for_claim}' is as stated in the provided capacity source(s)."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_matches_leaf,
        sources=capacity_urls_clean,
        additional_instruction=(
            "Confirm that the capacity page refers to the same venue identified and supports the stated capacity number. "
            "Allow common capacity variants (e.g., basketball vs. concert configuration) as long as the cited number is reasonable."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Matt Rife December 2025 Florida venue and capacity task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Gate capacity verification after venue/show identification
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_capacity(),
        template_class=VenueCapacityExtraction,
        extraction_name="extracted_venue_capacity",
    )

    # Add a compact custom info entry for convenience in summary
    evaluator.add_custom_info(
        info={
            "venue_name": extracted.venue_name,
            "venue_location_text": extracted.venue_location_text,
            "performance_date_text": extracted.performance_date_text,
            "stay_golden_tour_label": extracted.stay_golden_tour_label,
            "tour_date_urls": normalize_urls(extracted.tour_date_urls),
            "capacity_number_text": extracted.capacity_number_text,
            "capacity_urls": normalize_urls(extracted.capacity_urls),
            "parsed_capacity_int": parse_capacity_to_int(extracted.capacity_number_text),
        },
        info_type="extraction_overview",
        info_name="extraction_overview",
    )

    # Build verification tree and perform checks
    await build_verification_tree(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()