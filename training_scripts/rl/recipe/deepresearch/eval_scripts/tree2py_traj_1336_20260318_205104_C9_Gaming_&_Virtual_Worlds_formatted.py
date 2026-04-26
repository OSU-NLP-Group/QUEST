import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "summer2026_esports_premier_events"
TASK_DESCRIPTION = """I am researching the competitive esports landscape for the summer of 2026. Please identify three major esports tournaments that meet all of the following criteria:

1. The tournament must be for one of these games: Counter-Strike 2, Dota 2, or League of Legends.

2. The tournament must be a premier-tier competitive event, meaning it is officially designated as a Major, International, World Championship, or equivalent top-tier tournament recognized by the game's official competitive circuit.

3. The tournament must be scheduled to take place between May 1, 2026 and August 31, 2026 (inclusive). Both the start date and end date must fall within this window and must be publicly confirmed (not listed as "TBA" or "To Be Announced").

4. The tournament must have a confirmed total prize pool of at least $1,000,000 USD.

5. Each of the three tournaments must be held in a different country. No two tournaments can be in the same country.

6. The tournament must have confirmed venue information, including the specific city and the name of the arena or venue where it will be held.

7. Basic tournament format information must be publicly available, including the number of participating teams and the general tournament structure.

For each tournament, please provide:
- Tournament name
- Game title
- Start and end dates
- Prize pool amount
- Country, city, and venue name
- Brief format description (number of teams, basic structure)
- Reference URL from an official source (tournament organizer, Liquipedia, official game developer, or recognized esports database)
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TournamentItem(BaseModel):
    name: Optional[str] = None
    game: Optional[str] = None  # Normalize to "Counter-Strike 2", "Dota 2", or "League of Legends" when possible
    start_date: Optional[str] = None  # Full date string if present; null or "TBA" otherwise
    end_date: Optional[str] = None
    prize_pool: Optional[str] = None  # Keep as string (e.g., "$2,000,000", "USD 2M", "≥$1,000,000")
    country: Optional[str] = None
    city: Optional[str] = None
    venue: Optional[str] = None
    teams_count: Optional[str] = None  # e.g., "16", "24", or "24 teams"
    format_description: Optional[str] = None  # brief structure summary
    tier_or_designation: Optional[str] = None  # e.g., "Major", "The International", "World Championship"
    reference_urls: List[str] = Field(default_factory=list)  # official or recognized esports DB links


class TournamentExtraction(BaseModel):
    tournaments: List[TournamentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tournaments() -> str:
    return """
Extract all esports tournaments mentioned in the answer. For each tournament, extract the following fields exactly as stated in the answer (use null if a field is missing):

- name: Full tournament name as written.
- game: The game title. If possible, normalize to one of: "Counter-Strike 2", "Dota 2", "League of Legends" (accept common variants like "CS2" -> "Counter-Strike 2", "LoL" -> "League of Legends").
- start_date: The published start date string with year (e.g., "June 10, 2026"). Use null if not explicitly provided or listed as TBA.
- end_date: The published end date string with year. Use null if not explicitly provided or listed as TBA.
- prize_pool: The stated total prize pool (e.g., "$2,000,000", "USD 2M"). Use null if not explicitly provided or listed as TBA.
- country: Host country (use full country name if present). Use null if missing.
- city: Host city. Use null if missing.
- venue: Specific arena or venue name. Use null if missing.
- teams_count: The number of participating teams if stated (keep as string). Use null if missing.
- format_description: A brief description of the format/structure in plain text (e.g., "24 teams; group stage into playoffs"). Use null if missing.
- tier_or_designation: The top-tier designation if explicitly stated (e.g., "Major", "The International", "World Championship", etc.). Use null if missing.
- reference_urls: An array of all URLs cited for this tournament's facts. Include official sources (organizer, publisher), Liquipedia, or recognized esports databases. If none are present in the answer, return an empty array.

Return a JSON object with:
{
  "tournaments": [ { ... }, ... ]
}
Only extract what is explicitly in the answer; do not invent any missing details. If the answer lists more than 3 tournaments, include all of them; we will later consider only the first 3.
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
ALLOWED_GAMES = {"counter-strike 2", "cs2", "dota 2", "league of legends", "lol"}


def normalize_country(val: Optional[str]) -> Optional[str]:
    return val.strip().lower() if isinstance(val, str) else None


def has_sources(item: TournamentItem) -> bool:
    return bool(item.reference_urls and len(item.reference_urls) > 0)


# --------------------------------------------------------------------------- #
# Verification logic per tournament                                           #
# --------------------------------------------------------------------------- #
async def verify_single_tournament(evaluator: Evaluator, parent_node, item: TournamentItem, idx: int) -> None:
    """
    Build and verify the tree for one tournament according to the rubric leaves.
    """
    tid = idx + 1
    sources = item.reference_urls if item and item.reference_urls else []

    # Container for Tournament_i (non-critical; children are critical checks)
    t_node = evaluator.add_parallel(
        id=f"Tournament_{tid}",
        desc=f"{['First','Second','Third'][idx]} qualifying tournament meeting all criteria.",
        parent=parent_node,
        critical=False
    )

    # T{i}_Game (critical)
    n_game = evaluator.add_leaf(
        id=f"T{tid}_Game",
        desc="Tournament is for Counter-Strike 2, Dota 2, or League of Legends.",
        parent=t_node,
        critical=True
    )
    game_claim = (
        "This tournament is a competition for one of the following games: "
        "Counter-Strike 2 (CS2), Dota 2, or League of Legends (LoL)."
    )
    await evaluator.verify(
        claim=game_claim,
        node=n_game,
        sources=sources,
        additional_instruction="Rely on the page to confirm the game's title. Accept common synonyms like CS2 for Counter-Strike 2 and LoL for League of Legends."
    )

    # T{i}_Tier (critical)
    n_tier = evaluator.add_leaf(
        id=f"T{tid}_Tier",
        desc="Tournament is a premier-tier event (Major, International, World Championship, or equivalent).",
        parent=t_node,
        critical=True
    )
    tier_claim = (
        "This tournament is an official premier/top-tier event in its game's circuit "
        "(e.g., a CS2 Major, Dota 2's The International, or the League of Legends World Championship, "
        "or an equivalently designated premier-tier event)."
    )
    await evaluator.verify(
        claim=tier_claim,
        node=n_tier,
        sources=sources,
        additional_instruction="Look for explicit top-tier labels (Major, The International, World Championship) or equivalent premier designation recognized by the official circuit."
    )

    # T{i}_Dates (parallel, critical) with Start, End, Dates_URL
    n_dates = evaluator.add_parallel(
        id=f"T{tid}_Dates",
        desc="Tournament has confirmed start and end dates between May 1 and August 31, 2026.",
        parent=t_node,
        critical=True
    )
    # Start date
    n_start = evaluator.add_leaf(
        id=f"T{tid}_Start_Date",
        desc="Start date is confirmed and falls within May 1 to August 31, 2026.",
        parent=n_dates,
        critical=True
    )
    start_claim = (
        "The tournament's start date is publicly confirmed (not TBA) and falls between "
        "May 1, 2026 and August 31, 2026 inclusive."
    )
    await evaluator.verify(
        claim=start_claim,
        node=n_start,
        sources=sources,
        additional_instruction="Verify the exact start date on the page and ensure it lies within the specified 2026 summer window."
    )
    # End date
    n_end = evaluator.add_leaf(
        id=f"T{tid}_End_Date",
        desc="End date is confirmed and falls within May 1 to August 31, 2026.",
        parent=n_dates,
        critical=True
    )
    end_claim = (
        "The tournament's end date is publicly confirmed (not TBA) and falls between "
        "May 1, 2026 and August 31, 2026 inclusive."
    )
    await evaluator.verify(
        claim=end_claim,
        node=n_end,
        sources=sources,
        additional_instruction="Verify the exact end date on the page and ensure it lies within the specified 2026 summer window."
    )
    # Dates_URL (existence of source URL)
    evaluator.add_custom_node(
        result=has_sources(item),
        id=f"T{tid}_Dates_URL",
        desc="URL reference provided for date information from official source.",
        parent=n_dates,
        critical=True
    )

    # T{i}_Prize_Pool (parallel, critical) with Prize_Amount, Prize_URL
    n_prize = evaluator.add_parallel(
        id=f"T{tid}_Prize_Pool",
        desc="Tournament has a confirmed prize pool of at least $1,000,000 USD.",
        parent=t_node,
        critical=True
    )
    n_prize_amt = evaluator.add_leaf(
        id=f"T{tid}_Prize_Amount",
        desc="Prize pool amount is at least $1,000,000 USD.",
        parent=n_prize,
        critical=True
    )
    prize_claim = (
        "The tournament's total prize pool is publicly confirmed (not TBA) and is at least $1,000,000 USD "
        "or clearly above that amount (or equivalent value in other currencies)."
    )
    await evaluator.verify(
        claim=prize_claim,
        node=n_prize_amt,
        sources=sources,
        additional_instruction="Use the provided page to confirm the total prize pool and ensure it meets or exceeds $1,000,000 USD (allow reasonable currency conversions)."
    )
    evaluator.add_custom_node(
        result=has_sources(item),
        id=f"T{tid}_Prize_URL",
        desc="URL reference provided for prize pool information from official source.",
        parent=n_prize,
        critical=True
    )

    # T{i}_Location (parallel, critical) with Country, City, Venue, Location_URL
    n_loc = evaluator.add_parallel(
        id=f"T{tid}_Location",
        desc="Tournament has confirmed country, city, and venue information.",
        parent=t_node,
        critical=True
    )
    # Country
    n_country = evaluator.add_leaf(
        id=f"T{tid}_Country",
        desc="Country where tournament is held is specified.",
        parent=n_loc,
        critical=True
    )
    if item.country:
        country_claim = f"The tournament's host country is publicly confirmed (not TBA) and is '{item.country}'."
    else:
        country_claim = "The tournament's host country is publicly confirmed (not TBA)."
    await evaluator.verify(
        claim=country_claim,
        node=n_country,
        sources=sources,
        additional_instruction="Confirm that the page states the country explicitly; do not accept TBA."
    )
    # City
    n_city = evaluator.add_leaf(
        id=f"T{tid}_City",
        desc="City where tournament is held is specified.",
        parent=n_loc,
        critical=True
    )
    if item.city:
        city_claim = f"The tournament's host city is publicly confirmed (not TBA) and is '{item.city}'."
    else:
        city_claim = "The tournament's host city is publicly confirmed (not TBA)."
    await evaluator.verify(
        claim=city_claim,
        node=n_city,
        sources=sources,
        additional_instruction="Confirm the page provides a specific city; do not accept TBA."
    )
    # Venue
    n_venue = evaluator.add_leaf(
        id=f"T{tid}_Venue",
        desc="Specific venue name or arena is specified.",
        parent=n_loc,
        critical=True
    )
    if item.venue:
        venue_claim = f"The tournament's specific venue/arena is publicly confirmed (not TBA) and is '{item.venue}'."
    else:
        venue_claim = "The tournament's specific venue/arena is publicly confirmed (not TBA)."
    await evaluator.verify(
        claim=venue_claim,
        node=n_venue,
        sources=sources,
        additional_instruction="Confirm the page lists a specific named venue/arena; do not accept TBA."
    )
    evaluator.add_custom_node(
        result=has_sources(item),
        id=f"T{tid}_Location_URL",
        desc="URL reference provided for location information from official source.",
        parent=n_loc,
        critical=True
    )

    # T{i}_Format (parallel, critical) with Format_Details, Format_URL
    n_fmt = evaluator.add_parallel(
        id=f"T{tid}_Format",
        desc="Tournament format information is available.",
        parent=t_node,
        critical=True
    )
    n_fmt_details = evaluator.add_leaf(
        id=f"T{tid}_Format_Details",
        desc="Basic format information (team count, structure) is provided.",
        parent=n_fmt,
        critical=True
    )
    # Prefer including extracted numbers/description if present; otherwise generic presence check
    if item.teams_count or item.format_description:
        snippet_parts = []
        if item.teams_count:
            snippet_parts.append(f"{item.teams_count} teams")
        if item.format_description:
            snippet_parts.append(item.format_description)
        snippet = "; ".join(snippet_parts)
        fmt_claim = (
            f"The publicly available format includes the number of participating teams and the basic structure "
            f"(e.g., groups/swiss plus playoffs). Specifically, the page indicates: {snippet}."
        )
    else:
        fmt_claim = (
            "The publicly available format includes the number of participating teams and the basic structure "
            "(e.g., group stage/swiss and playoffs)."
        )
    await evaluator.verify(
        claim=fmt_claim,
        node=n_fmt_details,
        sources=sources,
        additional_instruction="Confirm the page states both: (1) number of teams and (2) general structure (groups/swiss + playoffs, etc.)."
    )
    evaluator.add_custom_node(
        result=has_sources(item),
        id=f"T{tid}_Format_URL",
        desc="URL reference provided for format information from official source.",
        parent=n_fmt,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer against the Summer 2026 premier esports tournaments rubric.
    """
    # 1) Initialize evaluator (Root as SEQUENTIAL to evaluate tournaments then uniqueness)
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

    # Record constraints as "ground truth"/requirements for transparency
    evaluator.add_ground_truth({
        "allowed_games": ["Counter-Strike 2", "Dota 2", "League of Legends"],
        "date_window_inclusive": ["2026-05-01", "2026-08-31"],
        "min_prize_pool_usd": "1000000",
        "country_uniqueness_required": True,
        "venue_required": True,
        "format_required": "team count + basic structure",
        "sources_required": "official/organizer/publisher or Liquipedia/recognized DB"
    }, gt_type="requirements")

    # 2) Extract tournaments from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_tournaments(),
        template_class=TournamentExtraction,
        extraction_name="tournaments_extraction"
    )

    # Normalize and keep only the first 3 tournaments (pad with placeholders if fewer)
    tournaments: List[TournamentItem] = (extracted.tournaments or [])[:3]
    while len(tournaments) < 3:
        tournaments.append(TournamentItem())

    # 3) Build "Tournaments" group under root (parallel, non-critical)
    tournaments_group = evaluator.add_parallel(
        id="Tournaments",
        desc="Identify three tournaments that meet all individual tournament criteria.",
        parent=root,
        critical=False
    )

    # 4) Verify each of the three tournaments
    for i in range(3):
        await verify_single_tournament(evaluator, tournaments_group, tournaments[i], i)

    # 5) Country uniqueness (critical)
    # This check is placed after tournaments (root is SEQUENTIAL). If tournaments group doesn't fully pass,
    # this uniqueness node will be skipped by the aggregation logic; otherwise it must pass.
    countries = [normalize_country(t.country) for t in tournaments]
    # All three must be present and mutually distinct
    uniqueness_ok = all(c is not None and c != "" for c in countries) and len(set(countries)) == 3
    evaluator.add_custom_node(
        result=uniqueness_ok,
        id="Country_Uniqueness",
        desc="All three tournaments must be held in different countries.",
        parent=root,
        critical=True
    )

    # 6) Return summary
    return evaluator.get_summary()