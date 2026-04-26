import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ewc_2026_cs2_coverage_planning"
TASK_DESCRIPTION = (
    "A gaming content creator wants to attend and cover the Counter-Strike 2 tournament at the Esports World Cup 2026. "
    "To plan their trip and coverage schedule, they need to determine: (1) which week (by week number) the Counter-Strike 2 tournament is scheduled, "
    "(2) the exact date range for that week, (3) the venue location (city and country), and (4) at least one other major esports title that is scheduled "
    "during the same week as Counter-Strike 2. Provide this information with supporting reference URLs from official sources."
)

EXPECTED_WEEK = "Week 7"
EXPECTED_DATES = "August 17-23, 2026"
EXPECTED_CITY = "Riyadh"
EXPECTED_COUNTRY = "Saudi Arabia"
ALLOWED_WEEK7_GAMES = ["Fortnite", "Trackmania", "Crossfire"]  # Accept minor naming variants (e.g., CrossFire)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CS2Timing(BaseModel):
    week_number: Optional[str] = None  # e.g., "Week 7", "7", "Week Seven"
    date_range: Optional[str] = None   # e.g., "August 17-23, 2026"
    timing_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    city: Optional[str] = None         # e.g., "Riyadh"
    country: Optional[str] = None      # e.g., "Saudi Arabia"
    venue_urls: List[str] = Field(default_factory=list)


class ConcurrentGame(BaseModel):
    game_name: Optional[str] = None    # e.g., "Fortnite", "Trackmania", "Crossfire"
    game_urls: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    timing: Optional[CS2Timing] = None
    venue: Optional[VenueInfo] = None
    concurrent_game: Optional[ConcurrentGame] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_planning_info() -> str:
    return """
    Extract from the answer the planning details for Esports World Cup 2026 (EWC 2026) Counter-Strike 2 (CS2) tournament.
    You must return a JSON with the following structure and fields:

    {
      "timing": {
        "week_number": string or null,       // the week number as stated in the answer, e.g., "Week 7", "7", "Week Seven"
        "date_range": string or null,        // the exact date range for that week as stated in the answer, e.g., "August 17-23, 2026"
        "timing_urls": string[]              // URL(s) in the answer that support the CS2 timing/week/dates; use only URLs explicitly present in the answer
      },
      "venue": {
        "city": string or null,              // city as stated in the answer, e.g., "Riyadh"
        "country": string or null,           // country as stated in the answer, e.g., "Saudi Arabia"
        "venue_urls": string[]               // URL(s) in the answer that support the venue/location; use only URLs explicitly present in the answer
      },
      "concurrent_game": {
        "game_name": string or null,         // one other major game scheduled the same week as CS2 as stated (e.g., "Fortnite", "Trackmania", or "Crossfire")
        "game_urls": string[]                // URL(s) in the answer that support this concurrent game scheduling; use only URLs explicitly present in the answer
      }
    }

    Rules:
    - Extract exactly what the answer states; do not infer or add information.
    - For URLs, include only actual URLs explicitly present in the answer (plain or markdown links).
    - If a field is missing in the answer, set it to null (or [] for URL arrays).
    - Do not normalize values; keep original casing and formatting from the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_timing_checks(evaluator: Evaluator, parent_node, extracted: PlanExtraction) -> None:
    """Build and verify the CS2 tournament timing subtree."""
    timing_node = evaluator.add_parallel(
        id="CS2_Tournament_Timing",
        desc="Verify correct identification of when the Counter-Strike 2 tournament occurs at EWC 2026",
        parent=parent_node,
        critical=True
    )

    week_str = (extracted.timing.week_number if extracted and extracted.timing else None) or ""
    dates_str = (extracted.timing.date_range if extracted and extracted.timing else None) or ""
    timing_urls = _safe_urls(extracted.timing.timing_urls if extracted and extracted.timing else [])

    # Timing reference URL presence (existence, critical)
    evaluator.add_custom_node(
        result=len(timing_urls) > 0,
        id="Timing_Reference_URL",
        desc="A valid reference URL supporting the CS2 tournament timing information is provided",
        parent=timing_node,
        critical=True
    )

    # Week number check (critical)
    week_leaf = evaluator.add_leaf(
        id="Week_Number",
        desc="The week number is correctly identified as Week 7",
        parent=timing_node,
        critical=True
    )
    week_claim = (
        f"The answer identifies the week number as '{week_str}'. "
        f"Treat '7', 'Week 7', 'Week Seven', or similar variants as equivalent to {EXPECTED_WEEK}. "
        f"Judge this claim as correct only if the provided value corresponds to {EXPECTED_WEEK}."
    )
    await evaluator.verify(
        claim=week_claim,
        node=week_leaf,
        additional_instruction=(
            "If the week number is missing or empty, mark as incorrect. Accept minor variations like 'Week Seven' or 'W7'."
        )
    )

    # Exact dates check (critical, needs sources)
    dates_leaf = evaluator.add_leaf(
        id="Exact_Dates",
        desc=f"The exact date range is correctly identified as {EXPECTED_DATES}",
        parent=timing_node,
        critical=True
    )
    dates_claim = (
        f"Esports World Cup 2026 Week 7 runs from {dates_str}. "
        f"Mark this as correct only if {dates_str} essentially matches '{EXPECTED_DATES}' "
        f"(allowing minor formatting variants like 'Aug 17–23, 2026'). "
        f"Use the provided sources to confirm Week 7's dates."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_leaf,
        sources=timing_urls,
        additional_instruction=(
            "Prefer official sources (e.g., esportsworldcup.com). If the answer's date range deviates from 'August 17-23, 2026', "
            "or sources are inconsistent/irrelevant, mark as incorrect."
        )
    )


async def build_venue_checks(evaluator: Evaluator, parent_node, extracted: PlanExtraction) -> None:
    """Build and verify the venue/location subtree."""
    venue_node = evaluator.add_parallel(
        id="Venue_Information",
        desc="Verify correct venue location details for the tournament",
        parent=parent_node,
        critical=True
    )

    city = (extracted.venue.city if extracted and extracted.venue else None) or ""
    country = (extracted.venue.country if extracted and extracted.venue else None) or ""
    venue_urls = _safe_urls(extracted.venue.venue_urls if extracted and extracted.venue else [])

    # Venue URL presence (existence, critical)
    evaluator.add_custom_node(
        result=len(venue_urls) > 0,
        id="Venue_Reference_URL",
        desc="A valid reference URL supporting the venue information is provided",
        parent=venue_node,
        critical=True
    )

    # Venue location correctness (critical, with source verification)
    venue_leaf = evaluator.add_leaf(
        id="Venue_Location",
        desc=f"The location is correctly identified as {EXPECTED_CITY}, {EXPECTED_COUNTRY}",
        parent=venue_node,
        critical=True
    )

    venue_claim = (
        f"Esports World Cup 2026 (including the Counter-Strike 2 tournament) takes place in {EXPECTED_CITY}, {EXPECTED_COUNTRY}. "
        f"The answer lists the location as '{city}, {country}'. "
        f"Mark this as correct only if that listing matches '{EXPECTED_CITY}, {EXPECTED_COUNTRY}'."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=venue_urls,
        additional_instruction=(
            "Prefer official EWC sources. Minor naming variants (e.g., 'KSA' for Saudi Arabia) are acceptable if unambiguous."
        )
    )


async def build_concurrent_game_checks(evaluator: Evaluator, parent_node, extracted: PlanExtraction) -> None:
    """Build and verify the concurrent game subtree."""
    concurrent_node = evaluator.add_parallel(
        id="Concurrent_Game",
        desc="Identify at least one other game scheduled during the same week as CS2",
        parent=parent_node,
        critical=True
    )

    game_name = (extracted.concurrent_game.game_name if extracted and extracted.concurrent_game else None) or ""
    game_urls = _safe_urls(extracted.concurrent_game.game_urls if extracted and extracted.concurrent_game else [])

    # Game URL presence (existence, critical)
    evaluator.add_custom_node(
        result=len(game_urls) > 0,
        id="Game_Reference_URL",
        desc="A valid reference URL supporting the concurrent game information is provided",
        parent=concurrent_node,
        critical=True
    )

    # Game name validity and scheduled in Week 7 (critical, verify with URLs)
    game_leaf = evaluator.add_leaf(
        id="Game_Name",
        desc="A valid game name from Week 7 lineup is provided (Fortnite, Trackmania, or Crossfire)",
        parent=concurrent_node,
        critical=True
    )

    allowed_list_str = ", ".join(ALLOWED_WEEK7_GAMES)
    game_claim = (
        f"The Esports World Cup 2026 Week 7 lineup includes {game_name}, and {game_name} is one of the following titles: "
        f"{allowed_list_str}. Mark as correct only if both conditions hold and the provided URLs support that this title is indeed in Week 7."
    )
    await evaluator.verify(
        claim=game_claim,
        node=game_leaf,
        sources=game_urls,
        additional_instruction=(
            "Allow minor naming variants (e.g., 'CrossFire' vs 'Crossfire'). "
            "If URLs are irrelevant/unsupported, mark as incorrect."
        )
    )


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
    Evaluate an answer for EWC 2026 CS2 coverage planning:
    - Week number for CS2 (expected: Week 7)
    - Exact date range for that week (expected: August 17-23, 2026)
    - Venue location (expected: Riyadh, Saudi Arabia)
    - At least one other major title in the same week (Fortnite, Trackmania, or Crossfire)
    All with supporting official reference URLs.
    """
    # Initialize evaluator (root is a non-critical aggregator by design)
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

    # Extract structured information from the answer
    extraction: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_planning_info(),
        template_class=PlanExtraction,
        extraction_name="planning_extraction",
    )

    # Add ground truth/context info for transparency (not used for scoring directly)
    evaluator.add_ground_truth({
        "expected_week": EXPECTED_WEEK,
        "expected_dates": EXPECTED_DATES,
        "expected_location": f"{EXPECTED_CITY}, {EXPECTED_COUNTRY}",
        "allowed_week7_games": ALLOWED_WEEK7_GAMES,
    })

    # Build top-level critical planning node to mirror rubric's root being critical
    planning_node = evaluator.add_parallel(
        id="EWC_2026_CS2_Coverage_Planning",
        desc="Verify complete and accurate planning information for attending Counter-Strike 2 tournament at Esports World Cup 2026",
        parent=root,
        critical=True
    )

    # Build subtrees (all critical as per rubric)
    await build_timing_checks(evaluator, planning_node, extraction)
    await build_venue_checks(evaluator, planning_node, extraction)
    await build_concurrent_game_checks(evaluator, planning_node, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()