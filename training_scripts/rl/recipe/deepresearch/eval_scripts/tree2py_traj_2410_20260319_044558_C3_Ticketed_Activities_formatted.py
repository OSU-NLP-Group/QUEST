import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "miami_dec2025_ko_round"
TASK_DESCRIPTION = """
A heavyweight professional boxing match took place in December 2025 at a venue in Miami, Florida. The venue is the home arena of an NBA team and has a seating capacity of under 25,000 for boxing events. This particular boxing event was a sell-out and generated the highest-grossing boxing gate in the venue's history. The fight was broadcast on Netflix and was scheduled for 8 rounds, with each round lasting 3 minutes, using 10 oz gloves. One of the participants was a YouTuber-turned-professional boxer who had previously fought at AT&T Stadium in Arlington, Texas in 2024 (where that event drew over 70,000 attendees). His opponent in this December 2025 match was a former two-time unified heavyweight champion. The fight ended by knockout before all 8 scheduled rounds were completed. In which round did the fight end?
"""
EXPECTED_KO_ROUND = 6


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FightExtraction(BaseModel):
    """Information about the identified December 2025 boxing match, as stated in the answer."""
    fight_name: Optional[str] = None
    date: Optional[str] = None
    month: Optional[str] = None
    year: Optional[str] = None
    venue: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    nba_home_team: Optional[str] = None
    boxing_capacity: Optional[str] = None
    sellout: Optional[str] = None  # keep as string to be lenient (e.g., "sell-out", "sold out", "yes")
    highest_grossing_gate_at_venue: Optional[str] = None
    broadcast: Optional[str] = None
    scheduled_rounds: Optional[str] = None
    round_duration_minutes: Optional[str] = None
    gloves_oz: Optional[str] = None
    weight_class: Optional[str] = None
    youtuber_boxer: Optional[str] = None
    prior_att_stadium_year: Optional[str] = None
    prior_att_stadium_city: Optional[str] = None
    prior_attendance: Optional[str] = None
    opponent_name: Optional[str] = None
    opponent_title: Optional[str] = None
    finish_method: Optional[str] = None
    ending_round: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fight() -> str:
    return """
    Identify the specific December 2025 heavyweight professional boxing match referenced in the answer and extract the following fields exactly as stated in the answer text. If a field is missing, return null (or empty list for arrays).

    Required fields:
    - fight_name: The event/fight name or the two participants (e.g., "Fighter A vs Fighter B") if given.
    - date: The full date as mentioned (e.g., "December 14, 2025").
    - month: The month (e.g., "December"), if explicitly mentioned.
    - year: The year (e.g., "2025"), if explicitly mentioned.
    - venue: The venue/arena name.
    - city: The city of the venue.
    - state: The state of the venue.
    - nba_home_team: The NBA team that calls the venue home (if provided).
    - boxing_capacity: The seating capacity for boxing, if stated (use the exact text/range/number given).
    - sellout: Whether the event was described as a "sell-out" (use the exact wording or "yes/no" if explicitly stated).
    - highest_grossing_gate_at_venue: If stated, capture the exact phrasing that this was the highest-grossing boxing gate at that venue.
    - broadcast: The broadcaster/streaming platform (e.g., "Netflix").
    - scheduled_rounds: The number of scheduled rounds (e.g., "8").
    - round_duration_minutes: The stated duration of each round in minutes (e.g., "3").
    - gloves_oz: The glove weight (e.g., "10 oz").
    - weight_class: The bout’s weight class (e.g., "heavyweight").
    - youtuber_boxer: The name of the YouTuber-turned-professional boxer (e.g., "Jake Paul" or "KSI").
    - prior_att_stadium_year: The stated year of the prior AT&T Stadium fight (should be "2024" if mentioned).
    - prior_att_stadium_city: The stated city of the AT&T Stadium (e.g., "Arlington, Texas") if mentioned.
    - prior_attendance: The stated attendance for that 2024 event (e.g., "over 70,000") if mentioned.
    - opponent_name: The opponent’s name (e.g., "Anthony Joshua").
    - opponent_title: The title designation (e.g., "former two-time unified heavyweight champion"), if mentioned.
    - finish_method: The result method (e.g., "knockout", "KO").
    - ending_round: The round in which the fight ended, as explicitly stated in the answer. Keep the original format if present (e.g., "6", "Round 6", "6th").
    - sources: A list of ALL URLs explicitly mentioned in the answer that support this December 2025 fight and its details (event page, venue page, news reports, broadcaster page, etc.). If no URLs are provided, return an empty array.

    Important:
    - Extract ONLY what is explicitly provided in the answer text. Do not infer or invent.
    - For URLs, include all valid full URLs you can find in the answer (plain links or markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
NUMBER_WORDS = {
    "one": 1, "first": 1,
    "two": 2, "second": 2,
    "three": 3, "third": 3,
    "four": 4, "fourth": 4,
    "five": 5, "fifth": 5,
    "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7,
    "eight": 8, "eighth": 8,
}


def parse_round_to_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    text = s.strip().lower()
    # Try direct digit extraction
    m = re.search(r"\b([1-8])\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Remove common suffixes (st, nd, rd, th)
    m = re.search(r"\b([1-8])(?:st|nd|rd|th)?\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Try number words
    for w, v in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", text):
            return v
    return None


def normalize_yes_no_flag(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    t = s.strip().lower()
    if any(x in t for x in ["yes", "sell-out", "sellout", "sold out", "sold-out", "true"]):
        return True
    if any(x in t for x in ["no", "false", "not sell", "wasn't", "was not"]):
        return False
    return None


def event_label_from_extraction(ex: FightExtraction) -> str:
    if ex.fight_name:
        return ex.fight_name
    if ex.city or ex.venue:
        loc = ", ".join([p for p in [ex.city, ex.state] if p])
        v = f" at {ex.venue}" if ex.venue else ""
        return f"the December 2025 heavyweight boxing match{v}{(' in ' + loc) if loc else ''}"
    return "the December 2025 heavyweight boxing match"


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_conditions_verification(
    evaluator: Evaluator,
    parent_node,
    ex: FightExtraction
) -> None:
    """
    Build and verify the 'Verify_Match_Satisfies_Stated_Conditions' parallel node and its leaf checks.
    """
    verify_node = evaluator.add_parallel(
        id="Verify_Match_Satisfies_Stated_Conditions",
        desc="Verify the identified match satisfies all conditions described in the prompt/constraints.",
        parent=parent_node,
        critical=True
    )

    urls = ex.sources or []
    event_label = event_label_from_extraction(ex)
    venue_label = ex.venue or "the event venue"

    claims_and_nodes = []

    # 1) Venue_Located_In_Miami_Florida
    node_1 = evaluator.add_leaf(
        id="Venue_Located_In_Miami_Florida",
        desc="The venue is located in Miami, Florida, United States.",
        parent=verify_node,
        critical=True
    )
    claim_1 = f"The venue for {event_label} is located in Miami, Florida, United States."
    claims_and_nodes.append((claim_1, urls, node_1, "Pass if the page(s) clearly indicate Miami, Florida as the venue location. Allow minor formatting variants."))

    # 2) Venue_Is_NBA_Home_Arena
    node_2 = evaluator.add_leaf(
        id="Venue_Is_NBA_Home_Arena",
        desc="The venue is the home arena of an NBA team.",
        parent=verify_node,
        critical=True
    )
    if ex.nba_home_team:
        claim_2 = f"{venue_label} is the home arena of the {ex.nba_home_team}, an NBA team."
    else:
        claim_2 = f"{venue_label} is the home arena of an NBA team."
    claims_and_nodes.append((claim_2, urls, node_2, "Look for confirmation the arena is home to an NBA team (e.g., Miami Heat)."))

    # 3) Venue_Boxing_Capacity_Under_25000
    node_3 = evaluator.add_leaf(
        id="Venue_Boxing_Capacity_Under_25000",
        desc="The venue has a seating capacity under 25,000 for boxing events.",
        parent=verify_node,
        critical=True
    )
    claim_3 = f"The seating capacity for boxing at {venue_label} is under 25,000."
    claims_and_nodes.append((claim_3, urls, node_3, "If a capacity range is given, consider typical boxing configuration; pass if clearly under 25,000."))

    # 4) Event_Occurred_In_December_2025
    node_4 = evaluator.add_leaf(
        id="Event_Occurred_In_December_2025",
        desc="The boxing event occurred in December 2025.",
        parent=verify_node,
        critical=True
    )
    claim_4 = f"{event_label} took place in December 2025."
    claims_and_nodes.append((claim_4, urls, node_4, "Verify the date (month/year) aligns with December 2025."))

    # 5) Event_Was_Sellout
    node_5 = evaluator.add_leaf(
        id="Event_Was_Sellout",
        desc="The event was a sell-out.",
        parent=verify_node,
        critical=True
    )
    claim_5 = f"{event_label} was a sell-out event."
    claims_and_nodes.append((claim_5, urls, node_5, "Look for 'sell-out', 'sold out', or equivalent phrasing on the sources."))

    # 6) Event_Highest_Grossing_Boxing_Gate_At_Venue
    node_6 = evaluator.add_leaf(
        id="Event_Highest_Grossing_Boxing_Gate_At_Venue",
        desc="The event generated the highest-grossing boxing gate in the venue's history.",
        parent=verify_node,
        critical=True
    )
    claim_6 = f"{event_label} generated the highest-grossing boxing gate in the history of {venue_label}."
    claims_and_nodes.append((claim_6, urls, node_6, "Confirm wording like 'highest-grossing boxing gate at this venue' or equivalent."))

    # 7) Broadcast_On_Netflix
    node_7 = evaluator.add_leaf(
        id="Broadcast_On_Netflix",
        desc="The fight was broadcast on Netflix.",
        parent=verify_node,
        critical=True
    )
    claim_7 = f"The fight {event_label} was broadcast on Netflix."
    claims_and_nodes.append((claim_7, urls, node_7, "Verify broadcaster/streaming platform is Netflix."))

    # 8) Scheduled_For_8_Rounds
    node_8 = evaluator.add_leaf(
        id="Scheduled_For_8_Rounds",
        desc="The fight was scheduled for 8 rounds.",
        parent=verify_node,
        critical=True
    )
    claim_8 = f"The bout {event_label} was scheduled for 8 rounds."
    claims_and_nodes.append((claim_8, urls, node_8, "Confirm the scheduled distance is 8 rounds (even if it ended earlier)."))

    # 9) Rounds_Last_3_Minutes
    node_9 = evaluator.add_leaf(
        id="Rounds_Last_3_Minutes",
        desc="Each round was 3 minutes long.",
        parent=verify_node,
        critical=True
    )
    claim_9 = f"Each round in {event_label} lasted 3 minutes."
    claims_and_nodes.append((claim_9, urls, node_9, "Standard men's professional rounds are 3 minutes; confirm the sources indicate 3-minute rounds."))

    # 10) Used_10oz_Gloves
    node_10 = evaluator.add_leaf(
        id="Used_10oz_Gloves",
        desc="The bout used 10 oz gloves.",
        parent=verify_node,
        critical=True
    )
    claim_10 = f"The bout {event_label} used 10-ounce gloves."
    claims_and_nodes.append((claim_10, urls, node_10, "Look for '10 oz' or '10-ounce' gloves in official info or credible reports."))

    # 11) Bout_Is_Heavyweight
    node_11 = evaluator.add_leaf(
        id="Bout_Is_Heavyweight",
        desc="The bout was classified as a heavyweight professional boxing match.",
        parent=verify_node,
        critical=True
    )
    claim_11 = f"{event_label} was a heavyweight professional boxing match."
    claims_and_nodes.append((claim_11, urls, node_11, "Verify the weight class is heavyweight."))

    # 12) Participant_Is_Youtuber_Turned_Pro_Boxer
    node_12 = evaluator.add_leaf(
        id="Participant_Is_Youtuber_Turned_Pro_Boxer",
        desc="One participant was a YouTuber-turned-professional boxer.",
        parent=verify_node,
        critical=True
    )
    if ex.youtuber_boxer:
        claim_12 = f"{ex.youtuber_boxer} is a YouTuber-turned-professional boxer and was one of the participants in {event_label}."
    else:
        claim_12 = f"One of the participants in {event_label} is a YouTuber-turned-professional boxer."
    claims_and_nodes.append((claim_12, urls, node_12, "Confirm the fighter's background as a YouTuber who turned professional boxer."))

    # 13) Participant_Fought_At_ATT_Stadium_Arlington_2024
    node_13 = evaluator.add_leaf(
        id="Participant_Fought_At_ATT_Stadium_Arlington_2024",
        desc="That YouTuber-turned-pro boxer previously fought at AT&T Stadium in Arlington, Texas in 2024.",
        parent=verify_node,
        critical=True
    )
    if ex.youtuber_boxer:
        claim_13 = f"In 2024, {ex.youtuber_boxer} fought at AT&T Stadium in Arlington, Texas."
    else:
        claim_13 = "In 2024, the YouTuber-turned-professional boxer fought at AT&T Stadium in Arlington, Texas."
    claims_and_nodes.append((claim_13, urls, node_13, "Verify a 2024 fight at AT&T Stadium (Arlington, TX) for that participant."))

    # 14) ATT_Stadium_Event_Drew_Over_70000
    node_14 = evaluator.add_leaf(
        id="ATT_Stadium_Event_Drew_Over_70000",
        desc="The 2024 AT&T Stadium event drew over 70,000 attendees.",
        parent=verify_node,
        critical=True
    )
    claim_14 = "That 2024 AT&T Stadium event drew over 70,000 attendees."
    claims_and_nodes.append((claim_14, urls, node_14, "Check reported attendance exceeds 70,000."))

    # 15) Opponent_Is_Former_Two_Time_Unified_Heavyweight_Champion
    node_15 = evaluator.add_leaf(
        id="Opponent_Is_Former_Two_Time_Unified_Heavyweight_Champion",
        desc="The opponent was a former two-time unified heavyweight champion.",
        parent=verify_node,
        critical=True
    )
    if ex.opponent_name:
        claim_15 = f"The opponent, {ex.opponent_name}, is a former two-time unified heavyweight champion."
    else:
        claim_15 = "The opponent is a former two-time unified heavyweight champion."
    claims_and_nodes.append((claim_15, urls, node_15, "Confirm the opponent held the unified heavyweight titles twice in the past."))

    # 16) Fight_Ended_By_Knockout
    node_16 = evaluator.add_leaf(
        id="Fight_Ended_By_Knockout",
        desc="The fight ended by knockout.",
        parent=verify_node,
        critical=True
    )
    claim_16 = f"{event_label} ended by knockout (KO)."
    claims_and_nodes.append((claim_16, urls, node_16, "The sources should state KO/knockout as the method of victory."))

    # 17) Knockout_Occurred_Before_8_Rounds_Completed
    node_17 = evaluator.add_leaf(
        id="Knockout_Occurred_Before_8_Rounds_Completed",
        desc="The knockout occurred before all 8 scheduled rounds were completed.",
        parent=verify_node,
        critical=True
    )
    claim_17 = f"The knockout in {event_label} occurred before all 8 scheduled rounds were completed."
    claims_and_nodes.append((claim_17, urls, node_17, "Confirm the KO happened before the scheduled 8 rounds were fully completed."))

    # Run all verifications (parallel)
    await evaluator.batch_verify(claims_and_nodes)


async def build_report_round_verification(
    evaluator: Evaluator,
    parent_node,
    ex: FightExtraction
) -> None:
    """
    Build and verify the 'Report_Ending_Round' sequential node and its leaf checks.
    """
    report_node = evaluator.add_sequential(
        id="Report_Ending_Round",
        desc="Provide the specific round number in which the fight ended (the knockout round).",
        parent=parent_node,
        critical=True
    )

    # 1) Round_Number_Provided (existence + range check)
    parsed_round = parse_round_to_int(ex.ending_round)
    round_provided_ok = parsed_round is not None and 1 <= parsed_round <= 8
    evaluator.add_custom_node(
        result=round_provided_ok,
        id="Round_Number_Provided",
        desc="The answer states a specific round number (an integer from 1 to 8).",
        parent=report_node,
        critical=True
    )

    # 2) Round_Number_Satisfies_Constraint (must be 6th as per constraints)
    evaluator.add_custom_node(
        result=(parsed_round == EXPECTED_KO_ROUND),
        id="Round_Number_Satisfies_Constraint",
        desc=f"The provided round number matches the required round specified by the constraints (i.e., the fight ended in the {EXPECTED_KO_ROUND}th round).",
        parent=report_node,
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
    Evaluate an answer for the 'Miami December 2025 KO round' task.
    """
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

    # Extraction
    ex: FightExtraction = await evaluator.extract(
        prompt=prompt_extract_fight(),
        template_class=FightExtraction,
        extraction_name="fight_extraction"
    )

    # Add ground truth info (expected KO round)
    evaluator.add_ground_truth(
        {"expected_ko_round": EXPECTED_KO_ROUND, "notes": "Per task constraints, the correct round is the 6th."},
        gt_type="expected_ko_round"
    )

    # Build critical root node mirroring rubric
    det_node = evaluator.add_sequential(
        id="Determine_Knockout_Round",
        desc="Identify the boxing match that satisfies the stated conditions and report the round in which it ended by knockout.",
        parent=root,
        critical=True
    )

    # First: verify all stated conditions (parallel)
    await build_conditions_verification(evaluator, det_node, ex)

    # Second: report ending round (sequential)
    await build_report_round_verification(evaluator, det_node, ex)

    # Return summary
    return evaluator.get_summary()