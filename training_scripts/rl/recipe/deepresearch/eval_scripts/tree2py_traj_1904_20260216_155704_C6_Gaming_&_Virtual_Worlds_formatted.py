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
TASK_ID = "major_esports_tournaments_2026"
TASK_DESCRIPTION = (
    "Identify four major esports tournaments (Jan 1–Aug 31, 2026) matching the stated constraints and provide "
    "required details with supporting sources."
)

WINDOW_START = "2026-01-01"
WINDOW_END = "2026-08-31"

ALLOWED_GAMES = [
    "Counter-Strike 2", "CS2",
    "Rainbow Six Siege", "R6", "R6S",
    "Dota 2", "Dota2"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TournamentEntry(BaseModel):
    tournament_name: Optional[str] = None
    game_title: Optional[str] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_country_region: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    prize_pool_amount: Optional[str] = None
    tier_classification: Optional[str] = None
    spectator_info: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TournamentsExtraction(BaseModel):
    tournaments: List[TournamentEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tournaments() -> str:
    return (
        "Extract tournament entries mentioned in the answer that correspond to major esports tournaments taking place "
        "between January 1, 2026 and August 31, 2026. For each tournament, extract the following fields exactly as "
        "stated in the answer:\n"
        "1. tournament_name: The name of the tournament.\n"
        "2. game_title: The primary competitive title for the event (e.g., Counter-Strike 2, Rainbow Six Siege, Dota 2).\n"
        "3. venue_name: The name of the physical venue where the event is held.\n"
        "4. venue_city: The city where the venue is located.\n"
        "5. venue_country_region: The country or region of the venue.\n"
        "6. start_date: The publicly announced start date.\n"
        "7. end_date: The publicly announced end date.\n"
        "8. prize_pool_amount: The total prize pool amount (include currency symbol or unit exactly as shown).\n"
        "9. tier_classification: The tournament tier classification (e.g., Tier 1, S-Tier, Major-level).\n"
        "10. spectator_info: Any information about spectator attendance, ticket availability, venue capacity, or "
        "attendance policies.\n"
        "11. sources: A list of URLs provided in the answer that support the details above (official pages, reliable "
        "esports news/tracking websites). Include only valid URLs. If presented as markdown links, extract the actual URL.\n\n"
        "If any field is missing for a tournament, return null for that field. If no sources are present for a "
        "tournament, return an empty array for sources.\n"
        "Return a JSON object with a 'tournaments' array of these entries. Extract ALL tournaments mentioned in the "
        "answer (do not infer or invent any)."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _sanitize_urls(urls: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.lower().startswith("http://") or u.lower().startswith("https://")):
            # If missing protocol, prepend http:// per extraction rules
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _allowed_games_instruction() -> str:
    return (
        "Allowed games: Counter-Strike 2 (CS2), Rainbow Six Siege, Dota 2. "
        "Treat common abbreviations as equivalent (e.g., CS2, R6, R6S, Dota2). "
        "Verify the primary competitive title for the event using the provided sources."
    )


def _date_window_instruction() -> str:
    return (
        f"Check that the publicly announced start and end dates fall between {WINDOW_START} and {WINDOW_END} (inclusive). "
        "Use the dates on the provided source pages. If dates are partially outside this window, treat as not meeting the constraint."
    )


def _tier_instruction() -> str:
    return (
        "Verify that the tournament is classified as Tier 1, S-Tier, or Major-level by the organizer or established "
        "esports tracking sites (e.g., Liquipedia, HLTV, Esports Charts). Minor variations like 'S Tier' or 'Major' "
        "should be considered equivalent."
    )


def _spectator_instruction() -> str:
    return (
        "Verify that the sources provide spectator-related information such as ticket availability/sales, venue capacity, "
        "or attendance policies. Look for words like 'tickets', 'capacity', 'attendance', 'spectators', 'box office'."
    )


# --------------------------------------------------------------------------- #
# Verification for a single tournament                                        #
# --------------------------------------------------------------------------- #
async def verify_tournament(
    evaluator: Evaluator,
    parent_node,
    t: TournamentEntry,
    index: int,
) -> None:
    node = evaluator.add_parallel(
        id=f"Tournament_{index + 1}",
        desc=f"Tournament #{index + 1} entry is evaluated for constraints and required fields.",
        parent=parent_node,
        critical=False,
    )

    # Prepare sources
    sources = _sanitize_urls(t.sources)

    # 1) Tournament Name Provided (existence check, critical)
    evaluator.add_custom_node(
        result=bool(t.tournament_name and t.tournament_name.strip()),
        id=f"Tournament_Name_Provided_T{index + 1}",
        desc="Tournament name is provided.",
        parent=node,
        critical=True,
    )

    # 2) Sources Links Provided (existence check, critical)
    evaluator.add_custom_node(
        result=len(sources) >= 1,
        id=f"Sources_Links_Provided_T{index + 1}",
        desc="At least one official page or reliable esports news/tracking link is provided supporting the stated details.",
        parent=node,
        critical=True,
    )

    # 3) Major Event Type (verify by urls, critical)
    leaf_major = evaluator.add_leaf(
        id=f"Major_Event_Type_T{index + 1}",
        desc="Event is a major esports tournament (not a gaming convention or expo, and not a qualifier-only event).",
        parent=node,
        critical=True,
    )
    claim_major = (
        f"The event '{t.tournament_name or ''}' is a major esports LAN tournament and not a convention/expo, "
        f"and not an online-only qualifier."
    )
    await evaluator.verify(
        claim=claim_major,
        node=leaf_major,
        sources=sources,
        additional_instruction=(
            "Confirm the nature of the event as a primary tournament. Reject events that are expos/conventions or solely "
            "qualifiers. It must be a standalone major tournament."
        ),
    )

    # 4) Game Title Allowed (verify by urls, critical)
    leaf_game = evaluator.add_leaf(
        id=f"Game_Title_Allowed_T{index + 1}",
        desc="Primary competitive title is stated and is Counter-Strike 2 (CS2), Rainbow Six Siege, or Dota 2.",
        parent=node,
        critical=True,
    )
    claim_game = (
        f"The primary competitive title for '{t.tournament_name or ''}' is '{t.game_title or ''}', "
        "and it is one of the allowed games (CS2, Rainbow Six Siege, Dota 2)."
    )
    await evaluator.verify(
        claim=claim_game,
        node=leaf_game,
        sources=sources,
        additional_instruction=_allowed_games_instruction(),
    )

    # 5) Dates Confirmed And In Window (verify by urls, critical)
    leaf_dates = evaluator.add_leaf(
        id=f"Dates_Confirmed_And_In_Window_T{index + 1}",
        desc="Publicly announced start and end dates are provided and fall between Jan 1, 2026 and Aug 31, 2026 (inclusive).",
        parent=node,
        critical=True,
    )
    claim_dates = (
        f"The event '{t.tournament_name or ''}' has publicly announced dates from '{t.start_date or ''}' to '{t.end_date or ''}', "
        f"and these dates fall between {WINDOW_START} and {WINDOW_END}, inclusive."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=leaf_dates,
        sources=sources,
        additional_instruction=_date_window_instruction(),
    )

    # 6) In-Person LAN (verify by urls, critical)
    leaf_lan = evaluator.add_leaf(
        id=f"In_Person_LAN_T{index + 1}",
        desc="Event is an in-person LAN held at a physical venue (not online-only).",
        parent=node,
        critical=True,
    )
    claim_lan = (
        f"The event '{t.tournament_name or ''}' is held in-person at a physical venue (LAN/offline), not online-only."
    )
    await evaluator.verify(
        claim=claim_lan,
        node=leaf_lan,
        sources=sources,
        additional_instruction="Look for 'LAN', 'offline', 'arena', 'stadium', or the presence of a physical venue.",
    )

    # 7) Venue Name Provided (verify by urls, critical)
    leaf_venue_name = evaluator.add_leaf(
        id=f"Venue_Name_Provided_T{index + 1}",
        desc="Venue name is provided (publicly available/confirmable via sources).",
        parent=node,
        critical=True,
    )
    claim_venue_name = (
        f"The venue name for '{t.tournament_name or ''}' is '{t.venue_name or ''}'."
    )
    await evaluator.verify(
        claim=claim_venue_name,
        node=leaf_venue_name,
        sources=sources,
        additional_instruction="Confirm the venue name from the provided sources.",
    )

    # 8) Venue Location Provided (verify by urls, critical)
    leaf_venue_loc = evaluator.add_leaf(
        id=f"Venue_Location_Provided_T{index + 1}",
        desc="Venue location is provided (city and country/region) (publicly available/confirmable via sources).",
        parent=node,
        critical=True,
    )
    loc_city = t.venue_city or ""
    loc_country = t.venue_country_region or ""
    claim_venue_loc = (
        f"The venue location for '{t.tournament_name or ''}' is '{loc_city}, {loc_country}'."
    )
    await evaluator.verify(
        claim=claim_venue_loc,
        node=leaf_venue_loc,
        sources=sources,
        additional_instruction="Confirm the venue city and country/region from the provided sources.",
    )

    # 9) Prize Pool Amount And Minimum (verify by urls, critical)
    leaf_prize = evaluator.add_leaf(
        id=f"Prize_Pool_Amount_And_Minimum_T{index + 1}",
        desc="Total prize pool amount is provided and is at least $250,000 USD.",
        parent=node,
        critical=True,
    )
    pp = t.prize_pool_amount or ""
    claim_prize = (
        f"The total prize pool for '{t.tournament_name or ''}' is '{pp}', and it is at least $250,000 USD."
    )
    await evaluator.verify(
        claim=claim_prize,
        node=leaf_prize,
        sources=sources,
        additional_instruction=(
            "Verify the total prize pool amount and confirm that it meets or exceeds $250,000 USD. "
            "If the amount is in a different currency, judge whether it's clearly equivalent or above $250,000 USD."
        ),
    )

    # 10) Tier Classification Allowed (verify by urls, critical)
    leaf_tier = evaluator.add_leaf(
        id=f"Tier_Classification_Allowed_T{index + 1}",
        desc="Tournament tier classification is provided and is Tier 1, S-Tier, or Major-level per organizer or established esports tracking website.",
        parent=node,
        critical=True,
    )
    tier_text = t.tier_classification or ""
    claim_tier = (
        f"The tournament '{t.tournament_name or ''}' is classified as '{tier_text}', which corresponds to Tier 1, S-Tier, or Major-level."
    )
    await evaluator.verify(
        claim=claim_tier,
        node=leaf_tier,
        sources=sources,
        additional_instruction=_tier_instruction(),
    )

    # 11) Spectator Info Included (verify by urls, critical)
    leaf_spec = evaluator.add_leaf(
        id=f"Spectator_Info_Included_T{index + 1}",
        desc="Spectator attendance information is provided (e.g., ticket availability, venue capacity, or attendance policies).",
        parent=node,
        critical=True,
    )
    spec_text = t.spectator_info or ""
    claim_spec = (
        f"The sources provide spectator-related information for '{t.tournament_name or ''}', such as tickets, capacity, or policies. "
        f"Provided detail: '{spec_text}'."
    )
    await evaluator.verify(
        claim=claim_spec,
        node=leaf_spec,
        sources=sources,
        additional_instruction=_spectator_instruction(),
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel aggregator
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

    # Extract tournaments from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_tournaments(),
        template_class=TournamentsExtraction,
        extraction_name="tournaments_extraction",
    )

    # Record some custom info about allowed games and date window
    evaluator.add_custom_info(
        info={"allowed_games": ["Counter-Strike 2 (CS2)", "Rainbow Six Siege", "Dota 2"],
              "date_window": {"start": WINDOW_START, "end": WINDOW_END}},
        info_type="constraints",
        info_name="task_constraints",
    )

    # Check count: must provide four tournaments (critical)
    evaluator.add_custom_node(
        result=len(extracted.tournaments) >= 4,
        id="Four_Tournaments_Provided",
        desc="Response provides four tournament entries (not fewer).",
        parent=root,
        critical=True,
    )

    # Limit to first 4 tournaments for detailed verification; pad with empty if fewer
    tournaments = list(extracted.tournaments[:4])
    while len(tournaments) < 4:
        tournaments.append(TournamentEntry())

    # Build verification nodes for four tournaments
    for i, t in enumerate(tournaments):
        await verify_tournament(evaluator, root, t, i)

    # Return final summary
    return evaluator.get_summary()