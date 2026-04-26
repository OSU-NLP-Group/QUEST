import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "zach_bryan_ca_stadium_two_consecutive_2026"
TASK_DESCRIPTION = """
Zach Bryan's 2026 'With Heaven On Tour' includes multiple stadium venues across the United States. Identify the stadium venue in California that hosts Zach Bryan for two consecutive concert dates during July or August 2026. Provide the venue's official name, the city where it is located, and the specific dates of both performances.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TourVenueInfo(BaseModel):
    """Extracted single venue and date pair from the agent's answer."""
    venue_official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    performance_date_1: Optional[str] = None
    performance_date_2: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tour_venue() -> str:
    return """
    Identify in the answer the California stadium venue that hosts Zach Bryan for two consecutive concert dates during July or August 2026 on the 'With Heaven On Tour'. If multiple venues are listed, select the first venue that satisfies ALL of the following:
    1) The venue is in California,
    2) The venue is a stadium, and
    3) It has exactly two consecutive performance dates in July or August 2026.

    Extract the following fields for that ONE venue:
    - venue_official_name: The venue’s official name, exactly as stated in the answer.
    - city: The city where the venue is located (as stated in the answer).
    - state: The state (e.g., "California" or "CA") if mentioned; otherwise null.
    - performance_date_1: The first performance date as a full calendar date including year (e.g., "August 7, 2026" or "2026-08-07").
    - performance_date_2: The second performance date as a full calendar date including year.
      If the answer uses a range format like "Aug 7–8, 2026", expand into two separate full dates (e.g., "August 7, 2026" and "August 8, 2026").
    - support_urls: All URLs explicitly cited in the answer that support the venue and schedule details
      (e.g., Zach Bryan's official tour schedule page, official venue page, or official ticketing pages like Ticketmaster/SeatGeek for these dates).
      Extract only actual URLs shown in the answer. If none are present, return an empty array.

    Return null for any field that is not present in the answer text. Do not invent information.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _strip_ordinals(s: str) -> str:
    """Remove ordinal suffixes from day numbers (e.g., '7th' -> '7')."""
    if not s:
        return s
    for suffix in ["st", "nd", "rd", "th"]:
        # Replace only occurrences after a digit
        s = s.replace(f" {suffix},", ",").replace(f"{suffix},", ",").replace(f" {suffix} ", " ").replace(f"{suffix} ", " ")
    return s


def parse_date_string(s: Optional[str]) -> Optional[date]:
    """Try to parse a human date string into a date object using common formats."""
    if not s:
        return None
    s_norm = _strip_ordinals(s).strip()

    patterns = [
        "%B %d, %Y",     # August 7, 2026
        "%b %d, %Y",     # Aug 7, 2026
        "%B %d %Y",      # August 7 2026
        "%b %d %Y",      # Aug 7 2026
        "%Y-%m-%d",      # 2026-08-07
        "%m/%d/%Y",      # 08/07/2026
        "%m-%d-%Y",      # 08-07-2026
    ]
    # Some answers might include weekday names; try removing common prefixes
    for remover in ["Monday, ", "Tuesday, ", "Wednesday, ", "Thursday, ", "Friday, ", "Saturday, ", "Sunday, "]:
        if s_norm.startswith(remover):
            s_norm = s_norm[len(remover):]

    for fmt in patterns:
        try:
            return datetime.strptime(s_norm, fmt).date()
        except Exception:
            continue

    # Fallback: try splitting by commas or hyphens and reassembling
    try:
        cleaned = s_norm.replace("–", "-").replace("—", "-")
        parts = [p.strip() for p in cleaned.replace(",", "").split()]
        # Expect like: "August 7 2026" or "Aug 7 2026"
        if len(parts) >= 3:
            maybe = " ".join(parts[:3])
            for fmt in ["%B %d %Y", "%b %d %Y"]:
                try:
                    return datetime.strptime(maybe, fmt).date()
                except Exception:
                    pass
    except Exception:
        pass

    return None


def is_consecutive_days(d1: Optional[date], d2: Optional[date]) -> bool:
    if not d1 or not d2:
        return False
    return abs((d2 - d1).days) == 1


def dates_in_july_or_aug_2026(d1: Optional[date], d2: Optional[date]) -> bool:
    if not d1 or not d2:
        return False
    months_ok = {7, 8}
    return (d1.year == 2026 and d2.year == 2026 and d1.month in months_ok and d2.month in months_ok)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    extracted: TourVenueInfo
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # Create a critical top-level node to mirror the JSON root being critical
    task_main = evaluator.add_parallel(
        id="task_main",
        desc="Identify the California stadium venue on Zach Bryan's 2026 'With Heaven On Tour' that hosts exactly two consecutive concert dates in July or August 2026, and provide the required details.",
        parent=root_node,
        critical=True
    )

    # ----------------------- Output Completeness (critical) -----------------------
    completeness_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Answer includes all required fields (venue official name, city, and both performance dates).",
        parent=task_main,
        critical=True
    )

    # Leaf: Provide official venue name
    provide_venue_node = evaluator.add_custom_node(
        result=bool(extracted.venue_official_name and extracted.venue_official_name.strip()),
        id="provide_official_venue_name",
        desc="Provide the venue's official name.",
        parent=completeness_node,
        critical=True
    )

    # Leaf: Provide city
    provide_city_node = evaluator.add_custom_node(
        result=bool(extracted.city and extracted.city.strip()),
        id="provide_city_location",
        desc="Provide the city where the venue is located.",
        parent=completeness_node,
        critical=True
    )

    # Leaf: Provide two full performance dates (including year)
    d1 = parse_date_string(extracted.performance_date_1)
    d2 = parse_date_string(extracted.performance_date_2)
    two_full_dates_node = evaluator.add_custom_node(
        result=bool(d1 and d2),
        id="provide_two_full_performance_dates",
        desc="Provide both specific performance dates as full calendar dates (including year).",
        parent=completeness_node,
        critical=True
    )

    # ----------------------- Constraint Validation (critical) --------------------
    constraints_node = evaluator.add_parallel(
        id="constraint_validation",
        desc="Provided venue and dates satisfy all tour and scheduling constraints.",
        parent=task_main,
        critical=True
    )

    # Prepare claims and nodes for URL-based verifications
    support_urls = extracted.support_urls if extracted.support_urls else []

    # 1) On official tour schedule
    on_schedule_leaf = evaluator.add_leaf(
        id="on_official_tour_schedule",
        desc="The identified venue and the two stated dates appear on Zach Bryan's 2026 'With Heaven On Tour' official schedule.",
        parent=constraints_node,
        critical=True
    )
    schedule_claim = (
        f"Zach Bryan's 'With Heaven On Tour' schedule lists two shows at {extracted.venue_official_name} in {extracted.city}, California "
        f"on {extracted.performance_date_1} and {extracted.performance_date_2}, and lists only these two dates at this venue."
    )

    # 2) Venue is stadium
    stadium_leaf = evaluator.add_leaf(
        id="venue_is_stadium",
        desc="The identified venue is a stadium (not another venue type).",
        parent=constraints_node,
        critical=True
    )
    stadium_claim = f"{extracted.venue_official_name} is classified as a stadium."

    # 3) Venue in California
    ca_leaf = evaluator.add_leaf(
        id="venue_in_california",
        desc="The identified venue is located in the state of California.",
        parent=constraints_node,
        critical=True
    )
    ca_claim = f"{extracted.venue_official_name} is located in {extracted.city}, California, United States."

    # Launch URL-based verifications (parallel)
    claims_and_sources = [
        (
            schedule_claim,
            support_urls,
            on_schedule_leaf,
            "Use the provided URLs (official tour page, official venue, or official ticketing pages). "
            "Pass only if the page(s) explicitly show two dates matching exactly the provided dates at the specified venue for 'With Heaven On Tour'. "
            "If a page lists more than two dates for this venue, the claim should be rejected."
        ),
        (
            stadium_claim,
            support_urls,
            stadium_leaf,
            "Confirm that the venue is a stadium (e.g., the official venue page or credible sources describe it as a stadium). "
            "Minor naming variants are acceptable; classification must clearly be 'stadium'."
        ),
        (
            ca_claim,
            support_urls,
            ca_leaf,
            "Verify that the venue's location is in the state of California. "
            "Accept city + CA/California indicators or official address placing it in California."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)

    # 4) Exactly two consecutive dates (custom logic check)
    consecutive_leaf = evaluator.add_custom_node(
        result=bool(d1 and d2 and is_consecutive_days(d1, d2)),
        id="exactly_two_consecutive_dates",
        desc="The venue hosts Zach Bryan on exactly two scheduled performance dates and those dates are consecutive calendar days.",
        parent=constraints_node,
        critical=True
    )

    # 5) Dates within July or August 2026 (custom logic check)
    july_aug_leaf = evaluator.add_custom_node(
        result=bool(d1 and d2 and dates_in_july_or_aug_2026(d1, d2)),
        id="dates_within_july_or_august_2026",
        desc="Both performance dates fall within July or August 2026.",
        parent=constraints_node,
        critical=True
    )

    # Add some custom info to aid debugging
    evaluator.add_custom_info(
        {
            "venue_official_name": extracted.venue_official_name,
            "city": extracted.city,
            "state": extracted.state,
            "performance_date_1": extracted.performance_date_1,
            "performance_date_2": extracted.performance_date_2,
            "parsed_date_1": d1.isoformat() if d1 else None,
            "parsed_date_2": d2.isoformat() if d2 else None,
            "support_urls_count": len(support_urls),
            "support_urls": support_urls,
        },
        info_type="extraction_debug",
        info_name="extracted_fields"
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
    Evaluate the agent's answer for the Zach Bryan California stadium consecutive dates task.
    """
    # Initialize evaluator (root is non-critical by framework; we add a critical child node as task main)
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

    # Extraction
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_tour_venue(),
        template_class=TourVenueInfo,
        extraction_name="tour_venue_info"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted_info)

    # Return structured summary
    return evaluator.get_summary()