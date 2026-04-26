import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "awards_q1_2026"
TASK_DESCRIPTION = (
    "Identify 4 major film or television industry awards ceremonies taking place in the United States between "
    "January 1, 2026 and March 15, 2026. For each ceremony, provide the following information: the official ceremony "
    "name and edition number (e.g., '98th Academy Awards'), the exact date of the ceremony, and the venue name and "
    "the city where it is being held."
)

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 3, 15)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CeremonyItem(BaseModel):
    """Information for a single awards ceremony."""
    official_name_and_edition: Optional[str] = None
    date: Optional[str] = None
    venue: Optional[str] = None
    city: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CeremoniesExtraction(BaseModel):
    """Container for multiple ceremonies extracted from the answer."""
    ceremonies: List[CeremonyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ceremonies() -> str:
    return """
    Extract up to 10 film or television industry awards ceremonies mentioned in the answer.
    For each ceremony, extract:
    1. official_name_and_edition: The official ceremony name together with the edition number if provided (e.g., "98th Academy Awards", "32nd SAG Awards"). If the edition number is not explicitly mentioned, still extract the name as given.
    2. date: The exact calendar date of the ceremony (e.g., "March 1, 2026", "2026-02-14").
    3. venue: The venue name (e.g., "Dolby Theatre", "Shrine Auditorium").
    4. city: The city where the ceremony is held (optionally include state abbreviations, e.g., "Los Angeles, CA").
    5. source_urls: Any URLs explicitly cited in the answer for that ceremony (optional; return an empty list if none are given).

    Rules:
    - Extract information exactly as presented in the answer; do not invent missing fields.
    - Use null for any field that is not provided; use [] for source_urls if none are present.
    - Preserve the original formatting of names and dates.

    Return a JSON object:
    { "ceremonies": [ { "official_name_and_edition": ..., "date": ..., "venue": ..., "city": ..., "source_urls": [...] }, ... ] }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_event_name(name: str) -> str:
    """Normalize event name for distinctness checking (basic normalization)."""
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    # remove ordinal suffixes and digits to reduce duplicate variants like "98th Academy Awards" vs "Academy Awards"
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"\d+", "", s)
    s = s.replace("the ", "").strip()
    return s


def parse_date_str(date_str: Optional[str]) -> Optional[datetime]:
    """Try to parse a variety of common date formats. Return None if parsing fails."""
    if not date_str or not date_str.strip():
        return None
    s = date_str.strip()
    patterns = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%B %d, %Y",  # March 1, 2026
        "%b %d, %Y",  # Mar 1, 2026
        "%d %B %Y",   # 1 March 2026
        "%d %b %Y",   # 1 Mar 2026
        "%m/%d/%Y",   # 03/01/2026
    ]
    for p in patterns:
        try:
            return datetime.strptime(s, p)
        except Exception:
            continue
    # Try to handle cases like "March 1 2026" (missing comma)
    try:
        return datetime.strptime(s, "%B %d %Y")
    except Exception:
        pass
    try:
        return datetime.strptime(s, "%b %d %Y")
    except Exception:
        pass
    return None


def is_date_in_range(date_str: Optional[str]) -> bool:
    """Check if the ceremony date lies within the required window [START_DATE, END_DATE]."""
    dt = parse_date_str(date_str)
    if dt is None:
        return False
    return START_DATE <= dt <= END_DATE


def has_edition_number(name: Optional[str]) -> bool:
    """Check that the ceremony string includes an edition number (i.e., at least one digit)."""
    if not name or not name.strip():
        return False
    return bool(re.search(r"\d", name))


def display_event_name(ceremony: CeremonyItem, index: int) -> str:
    """Choose a display name for claims."""
    if ceremony.official_name_and_edition and ceremony.official_name_and_edition.strip():
        return ceremony.official_name_and_edition.strip()
    return f"Ceremony #{index + 1}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_ceremony(
    evaluator: Evaluator,
    parent_node,
    ceremony: CeremonyItem,
    index: int,
) -> None:
    """
    Build verification checks for one ceremony.
    """
    # Parent ceremony node (non-critical to allow partial scoring per ceremony)
    ceremony_node = evaluator.add_parallel(
        id=f"ceremony_{index + 1}",
        desc=f"Ceremony {index + 1} meets all constraints and includes all required fields",
        parent=parent_node,
        critical=False,
    )

    # Existence checks (critical under ceremony)
    name_ok = has_edition_number(ceremony.official_name_and_edition)
    name_node = evaluator.add_custom_node(
        result=name_ok,
        id=f"c{index + 1}_official_name_and_edition",
        desc=f"Ceremony {index + 1} official ceremony name and edition number are provided",
        parent=ceremony_node,
        critical=True
    )

    date_provided_ok = parse_date_str(ceremony.date) is not None
    date_provided_node = evaluator.add_custom_node(
        result=date_provided_ok,
        id=f"c{index + 1}_exact_date_provided",
        desc=f"Ceremony {index + 1} exact ceremony date is provided",
        parent=ceremony_node,
        critical=True
    )

    venue_ok = bool(ceremony.venue and ceremony.venue.strip())
    venue_node = evaluator.add_custom_node(
        result=venue_ok,
        id=f"c{index + 1}_venue_name_provided",
        desc=f"Ceremony {index + 1} venue name is provided",
        parent=ceremony_node,
        critical=True
    )

    city_ok = bool(ceremony.city and ceremony.city.strip())
    city_node = evaluator.add_custom_node(
        result=city_ok,
        id=f"c{index + 1}_city_provided",
        desc=f"Ceremony {index + 1} city is provided",
        parent=ceremony_node,
        critical=True
    )

    # Date range check (critical)
    date_in_range_node = evaluator.add_custom_node(
        result=is_date_in_range(ceremony.date),
        id=f"c{index + 1}_date_in_range",
        desc=f"Ceremony {index + 1} occurs between 2026-01-01 and 2026-03-15 (inclusive)",
        parent=ceremony_node,
        critical=True
    )

    # Major award classification (critical; use LLM reasoning)
    major_award_leaf = evaluator.add_leaf(
        id=f"c{index + 1}_major_award",
        desc=f"Ceremony {index + 1} is a major film or television industry awards ceremony",
        parent=ceremony_node,
        critical=True
    )
    event_name = display_event_name(ceremony, index)
    major_award_claim = (
        f"'{event_name}' is a major film or television industry awards ceremony."
    )
    await evaluator.verify(
        claim=major_award_claim,
        node=major_award_leaf,
        additional_instruction=(
            "Judge whether this is one of the major US film/TV awards ceremonies (e.g., Academy Awards/Oscars, "
            "Golden Globes, Screen Actors Guild Awards, Directors Guild of America Awards, Producers Guild of America "
            "Awards, Critics Choice Awards, Independent Spirit Awards, Emmy Awards, People's Choice Awards). "
            "Regional critic circles or minor festivals should NOT be considered major industry awards."
        ),
        extra_prerequisites=[name_node]
    )

    # US location check (critical; use LLM reasoning)
    us_location_leaf = evaluator.add_leaf(
        id=f"c{index + 1}_us_location",
        desc=f"Ceremony {index + 1} takes place in the United States",
        parent=ceremony_node,
        critical=True
    )
    location_claim = (
        f"The ceremony takes place in the United States, at '{ceremony.venue or ''}' in '{ceremony.city or ''}'."
    )
    await evaluator.verify(
        claim=location_claim,
        node=us_location_leaf,
        additional_instruction=(
            "Determine whether the specified city (and venue) is located within the United States. "
            "Allow city strings like 'Los Angeles, CA', 'New York, NY', or other US city/state formats."
        ),
        extra_prerequisites=[city_node]
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
    Evaluate an answer for the 'awards_q1_2026' task.

    Note: The original rubric marked the root as critical, but obj_task_eval enforces that a critical parent must have
    all critical children. To allow partial credit per ceremony while maintaining a global critical gate for the
    count/distinctness requirement, we set the root as non-critical and keep 'global_count_and_distinctness' as
    a critical child, which will gate the overall score if it fails.
    """
    # Initialize evaluator (root as PARALLEL non-critical)
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

    # Record task window as ground truth info
    evaluator.add_ground_truth({
        "date_window_start": START_DATE.strftime("%Y-%m-%d"),
        "date_window_end": END_DATE.strftime("%Y-%m-%d"),
        "requirement": "Exactly 4 distinct major film/TV awards ceremonies in the US within the window"
    }, gt_type="task_requirements")

    # Extract ceremonies mentioned in the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_ceremonies(),
        template_class=CeremoniesExtraction,
        extraction_name="ceremonies_extraction"
    )

    # Select at most the first 4 ceremonies as required
    ceremonies: List[CeremonyItem] = (extraction.ceremonies or [])[:4]

    # Pad to exactly 4 to keep a consistent structure
    while len(ceremonies) < 4:
        ceremonies.append(CeremonyItem())

    # Global count and distinctness check (critical at root)
    non_empty_names = [c.official_name_and_edition for c in ceremonies if c.official_name_and_edition]
    distinct_keys = [normalize_event_name(n) for n in non_empty_names]
    distinct_count = len(set(distinct_keys))
    exactly_four_provided = len([c for c in ceremonies if c.official_name_and_edition and c.official_name_and_edition.strip()]) == 4
    global_ok = exactly_four_provided and distinct_count == 4

    evaluator.add_custom_node(
        result=global_ok,
        id="global_count_and_distinctness",
        desc="Exactly 4 distinct ceremonies are provided (no duplicates)",
        parent=root,
        critical=True
    )

    # Add ceremonies verification blocks
    for idx, ceremony in enumerate(ceremonies):
        await verify_ceremony(evaluator, root, ceremony, idx)

    # Add some custom info for debugging and traceability
    evaluator.add_custom_info({
        "selected_ceremonies": [
            {
                "official_name_and_edition": c.official_name_and_edition,
                "date": c.date,
                "venue": c.venue,
                "city": c.city,
                "source_urls": c.source_urls,
            }
            for c in ceremonies
        ],
        "exactly_four_provided": exactly_four_provided,
        "distinct_names_count": distinct_count
    }, info_type="selection_summary")

    # Return final evaluation summary
    return evaluator.get_summary()