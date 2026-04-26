import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

TASK_ID = "athletics_2025_compendium"
TASK_DESCRIPTION = (
    "During the 2025 college football season and recent coaching history, several significant athletic milestones were achieved in American education. "
    "Identify the following four educational programs or individuals that meet all the specified criteria:\n\n"
    "1) FCS Football Program: Winner of the 2025 FCS national championship (game played in January 2026) with the listed conditions; "
    "2) Ivy League Football Program: 2025 champion that earned Ivy’s first automatic FCS playoff bid and achieved the listed game/comeback details; "
    "3) Ohio High School Football Program: 2025 Division I champion, first title in school history, perfect season, beat a Cincinnati team in the final; "
    "4) College Basketball Coach: reached 900 wins in 2021, 25 NCAA tournament appearances, Final Fours at two schools in 1992 and 2010. "
    "For each, provide name, location, and supporting URLs."
)


# ----------------------------- Data Models --------------------------------- #

class FCSInfo(BaseModel):
    program_name: Optional[str] = None
    program_location: Optional[str] = None
    program_urls: List[str] = Field(default_factory=list)

    championship_date: Optional[str] = None  # e.g., "January 2026"
    championship_opponent: Optional[str] = None
    championship_final_score: Optional[str] = None
    championship_overtime: Optional[str] = None  # e.g., "overtime", "OT", "yes"
    championship_margin_one_point: Optional[str] = None  # e.g., "1 point", "one point", "yes"
    championship_urls: List[str] = Field(default_factory=list)

    ivy_opponent_name: Optional[str] = None
    ivy_opponent_membership: Optional[str] = None  # e.g., "Ivy League"
    playoff_round_vs_ivy: Optional[str] = None  # e.g., "Second Round", "Quarterfinal"
    playoff_result_vs_ivy: Optional[str] = None  # e.g., "defeated", "won"
    playoff_urls: List[str] = Field(default_factory=list)

    first_ivy_playoff_ever: Optional[str] = None  # e.g., "first-ever Ivy League FCS playoff appearance"
    eligibility_change_note: Optional[str] = None  # e.g., "Ivy began allowing postseason in 2024"
    historical_urls: List[str] = Field(default_factory=list)


class IvyInfo(BaseModel):
    program_name: Optional[str] = None
    program_location: Optional[str] = None
    program_urls: List[str] = Field(default_factory=list)

    championship_share: Optional[str] = None  # e.g., "won", "shared"
    championship_urls: List[str] = Field(default_factory=list)
    conference_record_2025: Optional[str] = None
    overall_record_2025: Optional[str] = None

    bid_type: Optional[str] = None  # e.g., "automatic"
    conference_first_auto_bid: Optional[str] = None  # e.g., "first automatic bid in Ivy history"
    qualification_urls: List[str] = Field(default_factory=list)

    finale_opponent_name: Optional[str] = None
    finale_opponent_status_unbeaten: Optional[str] = None  # e.g., "previously undefeated"
    finale_result_win: Optional[str] = None  # e.g., "won"
    finale_urls: List[str] = Field(default_factory=list)

    halftime_deficit_over_20: Optional[str] = None  # e.g., "trailed by 21", "down by >20"
    comeback_final_result_win: Optional[str] = None  # e.g., "won"
    comeback_final_score: Optional[str] = None
    comeback_urls: List[str] = Field(default_factory=list)

    second_round_elimination: Optional[str] = None  # e.g., "lost in second round"
    eliminated_by_team: Optional[str] = None
    exit_urls: List[str] = Field(default_factory=list)


class HSInfo(BaseModel):
    program_name: Optional[str] = None
    program_location: Optional[str] = None
    program_urls: List[str] = Field(default_factory=list)

    division: Optional[str] = None  # should be "Division I"
    state: Optional[str] = None  # should be "Ohio"
    championship_year: Optional[str] = None  # "2025"
    state_champ_urls: List[str] = Field(default_factory=list)

    first_title_ever: Optional[str] = None  # e.g., "first state championship"
    historical_urls: List[str] = Field(default_factory=list)

    final_record: Optional[str] = None  # e.g., "15-0"
    championship_opponent_name: Optional[str] = None
    championship_opponent_city: Optional[str] = None  # "Cincinnati"
    championship_final_score: Optional[str] = None
    season_urls: List[str] = Field(default_factory=list)


class CoachInfo(BaseModel):
    coach_name: Optional[str] = None
    current_status: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)

    win_total_900: Optional[str] = None  # e.g., "900 career wins"
    milestone_year: Optional[str] = None  # "2021"
    milestone_details: Optional[str] = None  # opponent/date details
    milestone_urls: List[str] = Field(default_factory=list)

    tournament_appearances_25: Optional[str] = None  # e.g., "25 NCAA tournament appearances"
    tournament_record: Optional[str] = None
    tournament_urls: List[str] = Field(default_factory=list)

    final_four_two_schools: Optional[str] = None  # e.g., "Final Fours at two schools"
    final_four_school_names: Optional[str] = None  # comma-separated list, if provided
    multiple_schools_urls: List[str] = Field(default_factory=list)

    final_four_1992: Optional[str] = None
    final_four_2010: Optional[str] = None
    years_urls: List[str] = Field(default_factory=list)


class AllExtraction(BaseModel):
    fcs: Optional[FCSInfo] = None
    ivy: Optional[IvyInfo] = None
    hs: Optional[HSInfo] = None
    coach: Optional[CoachInfo] = None


# --------------------------- Extraction Prompt ----------------------------- #

def prompt_extract_all() -> str:
    return (
        "Extract structured information for four items from the answer. For each item, return all explicitly stated details and all URLs (as full links):\n\n"
        "1) FCS Football Program (2025 season champion):\n"
        "- program_name, program_location (city, state), program_urls\n"
        "- championship_date (e.g., 'January 2026'), championship_opponent, championship_final_score,\n"
        "- championship_overtime (e.g., 'overtime', 'OT', 'yes'), championship_margin_one_point (e.g., '1 point', 'yes'), championship_urls\n"
        "- ivy_opponent_name, ivy_opponent_membership (should indicate Ivy League), playoff_round_vs_ivy (e.g., 'First Round', 'Quarterfinal'), playoff_result_vs_ivy (e.g., 'won', 'defeated'), playoff_urls\n"
        "- first_ivy_playoff_ever (statement in the answer), eligibility_change_note (if mentioned, e.g., 'Ivy began allowing postseason in 2024'), historical_urls\n\n"
        "2) Ivy League Football Program (2025 champion and first automatic bid):\n"
        "- program_name, program_location, program_urls\n"
        "- championship_share (e.g., 'won', 'shared'), championship_urls, conference_record_2025, overall_record_2025\n"
        "- bid_type (should be 'automatic'), conference_first_auto_bid (statement indicating first in Ivy history), qualification_urls\n"
        "- finale_opponent_name, finale_opponent_status_unbeaten (e.g., 'previously undefeated'), finale_result_win ('won'), finale_urls\n"
        "- halftime_deficit_over_20 (e.g., 'trailed by >20 at halftime'), comeback_final_result_win ('won'), comeback_final_score, comeback_urls\n"
        "- second_round_elimination (e.g., 'lost in second round'), eliminated_by_team, exit_urls\n\n"
        "3) Ohio High School Football Program (2025 Division I champion):\n"
        "- program_name, program_location, program_urls\n"
        "- division ('Division I'), state ('Ohio'), championship_year ('2025'), state_champ_urls\n"
        "- first_title_ever (statement indicating first in school history), historical_urls\n"
        "- final_record (e.g., '15-0'), championship_opponent_name, championship_opponent_city ('Cincinnati' if applicable), championship_final_score, season_urls\n\n"
        "4) College Basketball Coach:\n"
        "- coach_name, current_status, identification_urls\n"
        "- win_total_900 (statement indicating 900 wins), milestone_year ('2021'), milestone_details, milestone_urls\n"
        "- tournament_appearances_25 (statement indicating 25 appearances), tournament_record, tournament_urls\n"
        "- final_four_two_schools (statement indicating two different schools), final_four_school_names, multiple_schools_urls\n"
        "- final_four_1992 (statement), final_four_2010 (statement), years_urls\n\n"
        "Rules:\n"
        "- Extract only what appears in the answer; if a field is missing, use null.\n"
        "- For each URL field, extract all full URLs (including protocol), including markdown links.\n"
        "- Do not invent details; preserve wording where applicable (numbers can be stated as text)."
    )


# --------------------------- Helper Functions ------------------------------ #

def _use_sources(*lists: List[str]) -> List[str]:
    """Return the first non-empty sources list; otherwise empty list."""
    for lst in lists:
        if lst:
            return lst
    return []


def _safe_text(val: Optional[str], default_if_none: str = "") -> str:
    return val if val else default_if_none


# ------------------------- Verification Functions -------------------------- #

async def verify_fcs_program(evaluator: Evaluator, parent_node, fcs: Optional[FCSInfo]) -> None:
    fcs_node = evaluator.add_parallel(
        id="fcs_football_program",
        desc="Identify the FCS football program that won the 2025 national championship under specific conditions",
        parent=parent_node,
        critical=False
    )

    # Program identification group (critical)
    prog_id = evaluator.add_parallel(
        id="program_identification",
        desc="Provide the name and location of the FCS program",
        parent=fcs_node,
        critical=True
    )
    # Name leaf
    prog_name_leaf = evaluator.add_leaf(
        id="program_name",
        desc="Official name of the university program",
        parent=prog_id,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program identified is '{_safe_text(fcs.program_name, 'the program')}'. Confirm the correct team name.",
        node=prog_name_leaf,
        sources=_use_sources(fcs.program_urls, fcs.championship_urls),
        additional_instruction="Verify the official team/program name via the provided sources."
    )
    # Location leaf
    prog_loc_leaf = evaluator.add_leaf(
        id="program_location",
        desc="City and state where program is located",
        parent=prog_id,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program is located in {_safe_text(fcs.program_location, 'the stated location')} (city/state).",
        node=prog_loc_leaf,
        sources=_use_sources(fcs.program_urls, fcs.championship_urls),
        additional_instruction="Confirm the program location (city and state) on official or reputable sources."
    )
    # Program URL reference leaf
    prog_url_ref = evaluator.add_leaf(
        id="program_url_reference",
        desc="URL reference for program identification",
        parent=prog_id,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided sources confirm the identity and location of '{_safe_text(fcs.program_name, 'the program')}'.",
        node=prog_url_ref,
        sources=_use_sources(fcs.program_urls, fcs.championship_urls),
        additional_instruction="Rely on official program pages or major reputable outlets to confirm name and location."
    )

    # Championship conditions (critical)
    champ = evaluator.add_parallel(
        id="fcs_championship_win",
        desc="Program won the FCS national championship in 2025 season with specific game conditions",
        parent=fcs_node,
        critical=True
    )
    # Overtime requirement
    ot_leaf = evaluator.add_leaf(
        id="overtime_requirement",
        desc="Championship game went to overtime",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim="The 2025 FCS national championship game went to overtime.",
        node=ot_leaf,
        sources=_use_sources(fcs.championship_urls),
        additional_instruction="Look for mentions of 'overtime' or 'OT' in the game recap or box score."
    )
    # Margin requirement
    margin_leaf = evaluator.add_leaf(
        id="margin_requirement",
        desc="Winning margin was exactly 1 point",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim="The winning margin in the 2025 FCS national title game was exactly 1 point.",
        node=margin_leaf,
        sources=_use_sources(fcs.championship_urls),
        additional_instruction="Confirm the final score reflects a one-point margin."
    )
    # Date requirement
    date_leaf = evaluator.add_leaf(
        id="championship_date",
        desc="Championship game was played in January 2026",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim="The FCS championship game for the 2025 season was played in January 2026.",
        node=date_leaf,
        sources=_use_sources(fcs.championship_urls),
        additional_instruction="Check the game date on the official recap or schedule page."
    )
    # Championship URL reference
    champ_url_leaf = evaluator.add_leaf(
        id="championship_url_reference",
        desc="URL reference confirming championship details",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim=f"The sources confirm that {_safe_text(fcs.program_name, 'the program')} won the 2025 FCS national championship and provide score/context.",
        node=champ_url_leaf,
        sources=_use_sources(fcs.championship_urls),
        additional_instruction="Verify winner, score, overtime, and date on reputable sources."
    )

    # Playoff path (critical)
    path = evaluator.add_parallel(
        id="playoff_path",
        desc="Program defeated an Ivy League football program during playoffs",
        parent=fcs_node,
        critical=True
    )
    ivy_member_leaf = evaluator.add_leaf(
        id="ivy_conference_membership",
        desc="Opponent was member of Ivy League conference",
        parent=path,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_safe_text(fcs.ivy_opponent_name, 'The opponent')} is a member of the Ivy League.",
        node=ivy_member_leaf,
        sources=_use_sources(fcs.playoff_urls, fcs.historical_urls),
        additional_instruction="Confirm conference affiliation of the named opponent."
    )
    playoff_result_leaf = evaluator.add_leaf(
        id="playoff_game_result",
        desc="Program defeated the Ivy League opponent",
        parent=path,
        critical=True
    )
    await evaluator.verify(
        claim=f"During the 2025 FCS playoffs, {_safe_text(fcs.program_name, 'the program')} defeated {_safe_text(fcs.ivy_opponent_name, 'the Ivy League opponent')}.",
        node=playoff_result_leaf,
        sources=_use_sources(fcs.playoff_urls),
        additional_instruction="Use game recap/box score pages for confirmation."
    )
    playoff_round_leaf = evaluator.add_leaf(
        id="playoff_round",
        desc="Game occurred during FCS playoff bracket",
        parent=path,
        critical=True
    )
    await evaluator.verify(
        claim=f"The game vs {_safe_text(fcs.ivy_opponent_name, 'the Ivy opponent')} occurred in the {_safe_text(fcs.playoff_round_vs_ivy, 'FCS playoffs')} of the FCS playoffs.",
        node=playoff_round_leaf,
        sources=_use_sources(fcs.playoff_urls),
        additional_instruction="Confirm the playoff round (first round, second round, quarterfinal, etc.)."
    )
    ivy_ref_leaf = evaluator.add_leaf(
        id="ivy_opponent_url_reference",
        desc="URL reference confirming Ivy League opponent details",
        parent=path,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm the Ivy League opponent identity and the playoff game details.",
        node=ivy_ref_leaf,
        sources=_use_sources(fcs.playoff_urls),
        additional_instruction="Cross-check conference membership and game outcome."
    )

    # Historical significance (critical core + non-critical extra)
    hist_core = evaluator.add_parallel(
        id="historical_significance",
        desc="The Ivy League opponent was making historic first appearance",
        parent=fcs_node,
        critical=True
    )
    first_playoff_leaf = evaluator.add_leaf(
        id="first_ivy_playoff",
        desc="The Ivy League opponent was making the conference's first-ever FCS playoff appearance",
        parent=hist_core,
        critical=True
    )
    await evaluator.verify(
        claim="This opponent was part of the Ivy League's first-ever FCS playoff appearance.",
        node=first_playoff_leaf,
        sources=_use_sources(fcs.historical_urls, fcs.playoff_urls),
        additional_instruction="Find explicit mention of 'first-ever' Ivy League appearance in the FCS playoffs."
    )
    hist_ref_leaf = evaluator.add_leaf(
        id="historical_url_reference",
        desc="URL reference confirming historical significance",
        parent=hist_core,
        critical=True
    )
    await evaluator.verify(
        claim="The sources explicitly confirm the historic first playoff appearance for the Ivy League.",
        node=hist_ref_leaf,
        sources=_use_sources(fcs.historical_urls),
        additional_instruction="Use conference releases or reputable news confirming the milestone."
    )
    # Non-critical additional historical note (placed under a separate non-critical node to satisfy critical-child consistency)
    hist_extra = evaluator.add_parallel(
        id="historical_significance_additional",
        desc="Additional historical context",
        parent=fcs_node,
        critical=False
    )
    eligibility_leaf = evaluator.add_leaf(
        id="playoff_eligibility_change",
        desc="Ivy League began allowing postseason play in 2024",
        parent=hist_extra,
        critical=False
    )
    await evaluator.verify(
        claim="The Ivy League began allowing postseason play in 2024.",
        node=eligibility_leaf,
        sources=_use_sources(fcs.historical_urls),
        additional_instruction="Confirm the policy change year via official Ivy League or NCAA sources."
    )


async def verify_ivy_program(evaluator: Evaluator, parent_node, ivy: Optional[IvyInfo]) -> None:
    ivy_node = evaluator.add_parallel(
        id="ivy_league_program",
        desc="Identify the Ivy League football program that earned the conference's first FCS playoff bid",
        parent=parent_node,
        critical=False
    )

    # Program identification (non-specified in rubric root text but required by task)
    prog_id = evaluator.add_parallel(
        id="ivy_program_identification",
        desc="Provide the name and location of the Ivy League program",
        parent=ivy_node,
        critical=True
    )
    ivy_name_leaf = evaluator.add_leaf(
        id="ivy_program_name",
        desc="Full name of the Ivy League football program",
        parent=prog_id,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program identified is '{_safe_text(ivy.program_name, 'the program')}'.",
        node=ivy_name_leaf,
        sources=_use_sources(ivy.program_urls, ivy.championship_urls),
        additional_instruction="Confirm the program's official name via reputable sources."
    )
    ivy_loc_leaf = evaluator.add_leaf(
        id="ivy_program_location",
        desc="Location (city/state) of the Ivy League program",
        parent=prog_id,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program is located in {_safe_text(ivy.program_location, 'the stated location')} (city/state).",
        node=ivy_loc_leaf,
        sources=_use_sources(ivy.program_urls, ivy.championship_urls),
        additional_instruction="Verify location via official program pages or reputable outlets."
    )
    ivy_id_ref = evaluator.add_leaf(
        id="ivy_identification_url_reference",
        desc="URL reference confirming Ivy program identification",
        parent=prog_id,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the program's identity and location.",
        node=ivy_id_ref,
        sources=_use_sources(ivy.program_urls, ivy.championship_urls),
        additional_instruction="Use official or reputable publications for confirmation."
    )

    # Conference championship (critical core)
    champ = evaluator.add_parallel(
        id="conference_championship",
        desc="Program won share of 2025 Ivy League championship",
        parent=ivy_node,
        critical=True
    )
    champ_share_leaf = evaluator.add_leaf(
        id="championship_share",
        desc="Won or shared the Ivy League title in 2025",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim="The program won or shared the Ivy League championship in 2025.",
        node=champ_share_leaf,
        sources=_use_sources(ivy.championship_urls),
        additional_instruction="Confirm 2025 Ivy standings and champion shares."
    )
    champ_url_leaf = evaluator.add_leaf(
        id="championship_url_reference",
        desc="URL reference confirming Ivy League championship",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the program was an Ivy League champion in 2025.",
        node=champ_url_leaf,
        sources=_use_sources(ivy.championship_urls),
        additional_instruction="Use Ivy League official site or reputable news coverage."
    )
    # Non-critical records, placed separately
    champ_details = evaluator.add_parallel(
        id="championship_details",
        desc="Additional championship records (non-critical)",
        parent=ivy_node,
        critical=False
    )
    conf_rec_leaf = evaluator.add_leaf(
        id="conference_record",
        desc="Provide conference record for 2025 season",
        parent=champ_details,
        critical=False
    )
    await evaluator.verify(
        claim=f"The program's 2025 Ivy conference record was {_safe_text(ivy.conference_record_2025, 'stated')}.",
        node=conf_rec_leaf,
        sources=_use_sources(ivy.championship_urls),
        additional_instruction="Confirm the exact Ivy record if listed."
    )
    overall_rec_leaf = evaluator.add_leaf(
        id="overall_record",
        desc="Provide overall record for 2025 season",
        parent=champ_details,
        critical=False
    )
    await evaluator.verify(
        claim=f"The program's 2025 overall record was {_safe_text(ivy.overall_record_2025, 'stated')}.",
        node=overall_rec_leaf,
        sources=_use_sources(ivy.championship_urls),
        additional_instruction="Confirm overall record via season summary pages."
    )

    # Playoff qualification (critical)
    qual = evaluator.add_parallel(
        id="playoff_qualification",
        desc="Received automatic bid to FCS playoffs as Ivy champion",
        parent=ivy_node,
        critical=True
    )
    bid_leaf = evaluator.add_leaf(
        id="bid_type",
        desc="Bid was automatic (not at-large)",
        parent=qual,
        critical=True
    )
    await evaluator.verify(
        claim="The program received an automatic bid to the FCS playoffs.",
        node=bid_leaf,
        sources=_use_sources(ivy.qualification_urls),
        additional_instruction="Confirm auto-bid status in the playoff qualification context."
    )
    first_auto_leaf = evaluator.add_leaf(
        id="conference_first",
        desc="First automatic bid in Ivy League history",
        parent=qual,
        critical=True
    )
    await evaluator.verify(
        claim="This was the first automatic FCS playoff bid in Ivy League history.",
        node=first_auto_leaf,
        sources=_use_sources(ivy.qualification_urls, ivy.championship_urls),
        additional_instruction="Look for 'first automatic bid' phrasing in official or reputable sources."
    )
    qual_url_leaf = evaluator.add_leaf(
        id="qualification_url_reference",
        desc="URL reference confirming playoff qualification",
        parent=qual,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the automatic bid and its historical first for the Ivy League.",
        node=qual_url_leaf,
        sources=_use_sources(ivy.qualification_urls),
        additional_instruction="Use official Ivy or NCAA materials when available."
    )

    # Regular season finale (critical)
    finale = evaluator.add_parallel(
        id="regular_season_finale",
        desc="Final regular season game circumstances",
        parent=ivy_node,
        critical=True
    )
    opponent_status_leaf = evaluator.add_leaf(
        id="opponent_status",
        desc="Final game opponent was previously undefeated",
        parent=finale,
        critical=True
    )
    await evaluator.verify(
        claim=f"The final regular season opponent ({_safe_text(ivy.finale_opponent_name, 'the opponent')}) was previously undefeated.",
        node=opponent_status_leaf,
        sources=_use_sources(ivy.finale_urls),
        additional_instruction="Confirm the opponent's undefeated status prior to the finale."
    )
    finale_win_leaf = evaluator.add_leaf(
        id="game_result",
        desc="Won the final regular season game",
        parent=finale,
        critical=True
    )
    await evaluator.verify(
        claim="The program won the final regular season game.",
        node=finale_win_leaf,
        sources=_use_sources(ivy.finale_urls),
        additional_instruction="Use game recap or summary confirming the win."
    )
    finale_ref_leaf = evaluator.add_leaf(
        id="finale_url_reference",
        desc="URL reference confirming regular season finale details",
        parent=finale,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the opponent's status and the finale victory.",
        node=finale_ref_leaf,
        sources=_use_sources(ivy.finale_urls),
        additional_instruction="Check official or reputable game summaries."
    )
    # Optional significance (non-critical)
    finale_extra = evaluator.add_parallel(
        id="finale_details",
        desc="Finale significance details (non-critical)",
        parent=ivy_node,
        critical=False
    )
    game_sig_leaf = evaluator.add_leaf(
        id="game_significance",
        desc="Victory clinched playoff berth and championship share",
        parent=finale_extra,
        critical=False
    )
    await evaluator.verify(
        claim="The finale victory clinched the playoff berth and championship share.",
        node=game_sig_leaf,
        sources=_use_sources(ivy.finale_urls, ivy.championship_urls),
        additional_instruction="Confirm any mention of clinching scenarios tied to the finale."
    )

    # First-round performance (sequential: comeback then exit)
    fr_perf = evaluator.add_sequential(
        id="first_round_performance",
        desc="First-round playoff game featured dramatic comeback",
        parent=ivy_node,
        critical=True
    )
    comeback = evaluator.add_parallel(
        id="comeback_victory",
        desc="Won first-round game after trailing by more than 20 points at halftime",
        parent=fr_perf,
        critical=True
    )
    halftime_leaf = evaluator.add_leaf(
        id="halftime_deficit",
        desc="Trailed by more than 20 points at halftime",
        parent=comeback,
        critical=True
    )
    await evaluator.verify(
        claim="The team trailed by more than 20 points at halftime in the first-round playoff game.",
        node=halftime_leaf,
        sources=_use_sources(ivy.comeback_urls),
        additional_instruction="Check halftime score or narrative confirming >20 deficit."
    )
    comeback_win_leaf = evaluator.add_leaf(
        id="final_result",
        desc="Won the game in comeback fashion",
        parent=comeback,
        critical=True
    )
    await evaluator.verify(
        claim="The team won the first-round game despite the large halftime deficit.",
        node=comeback_win_leaf,
        sources=_use_sources(ivy.comeback_urls),
        additional_instruction="Confirm final result and comeback context."
    )
    comeback_ref_leaf = evaluator.add_leaf(
        id="comeback_url_reference",
        desc="URL reference confirming comeback details",
        parent=comeback,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the halftime deficit and the comeback victory.",
        node=comeback_ref_leaf,
        sources=_use_sources(ivy.comeback_urls),
        additional_instruction="Use game recaps or reputable summaries confirming both points."
    )
    # Non-critical margin details placed separately
    comeback_extra = evaluator.add_parallel(
        id="comeback_margin_details",
        desc="Halftime deficit and final score details (non-critical)",
        parent=ivy_node,
        critical=False
    )
    margin_detail_leaf = evaluator.add_leaf(
        id="comeback_margin",
        desc="Provide halftime deficit and final score",
        parent=comeback_extra,
        critical=False
    )
    await evaluator.verify(
        claim=f"Halftime deficit: {_safe_text(ivy.halftime_deficit_over_20, 'stated')}; final score: {_safe_text(ivy.comeback_final_score, 'stated')}.",
        node=margin_detail_leaf,
        sources=_use_sources(ivy.comeback_urls),
        additional_instruction="Confirm the specific numbers if present."
    )

    # Second round exit (critical, sequential second)
    exit_grp = evaluator.add_parallel(
        id="second_round_exit",
        desc="Lost in second round of playoffs",
        parent=fr_perf,
        critical=True
    )
    eliminated_leaf = evaluator.add_leaf(
        id="elimination_game",
        desc="Program was eliminated in second round",
        parent=exit_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The program was eliminated in the second round of the FCS playoffs.",
        node=eliminated_leaf,
        sources=_use_sources(ivy.exit_urls),
        additional_instruction="Confirm the round and result via playoff bracket or recap."
    )
    exit_ref_leaf = evaluator.add_leaf(
        id="exit_url_reference",
        desc="URL reference confirming second-round result",
        parent=exit_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the second-round elimination.",
        node=exit_ref_leaf,
        sources=_use_sources(ivy.exit_urls),
        additional_instruction="Use reputable coverage or official bracket pages."
    )
    # Non-critical opponent identity
    exit_extra = evaluator.add_parallel(
        id="exit_details",
        desc="Second-round opponent identity (non-critical)",
        parent=ivy_node,
        critical=False
    )
    opp_id_leaf = evaluator.add_leaf(
        id="opponent_identity",
        desc="Provide name of team that eliminated program",
        parent=exit_extra,
        critical=False
    )
    await evaluator.verify(
        claim=f"The team that eliminated the program in the second round was {_safe_text(ivy.eliminated_by_team, 'stated')}.",
        node=opp_id_leaf,
        sources=_use_sources(ivy.exit_urls),
        additional_instruction="Confirm the opponent's name in the elimination game."
    )


async def verify_high_school_program(evaluator: Evaluator, parent_node, hs: Optional[HSInfo]) -> None:
    hs_node = evaluator.add_parallel(
        id="high_school_program",
        desc="Identify the Ohio high school football program that won its first Division I state championship in 2025",
        parent=parent_node,
        critical=False
    )

    # Identification
    prog_id = evaluator.add_parallel(
        id="hs_program_identification",
        desc="Provide the name and location of the high school program",
        parent=hs_node,
        critical=True
    )
    hs_name_leaf = evaluator.add_leaf(
        id="hs_program_name",
        desc="High school program name",
        parent=prog_id,
        critical=True
    )
    await evaluator.verify(
        claim=f"The high school football program is '{_safe_text(hs.program_name, 'the program')}'.",
        node=hs_name_leaf,
        sources=_use_sources(hs.program_urls, hs.state_champ_urls),
        additional_instruction="Confirm the official school/team name via OHSAA or reputable local coverage."
    )
    hs_loc_leaf = evaluator.add_leaf(
        id="hs_program_location",
        desc="High school program location (city/state)",
        parent=prog_id,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program is located in {_safe_text(hs.program_location, 'the stated location')} (city/state).",
        node=hs_loc_leaf,
        sources=_use_sources(hs.program_urls, hs.state_champ_urls),
        additional_instruction="Verify the location via school site or OHSAA."
    )
    hs_id_ref = evaluator.add_leaf(
        id="hs_program_url_reference",
        desc="URL reference for program identification",
        parent=prog_id,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the program's identity and location.",
        node=hs_id_ref,
        sources=_use_sources(hs.program_urls, hs.state_champ_urls),
        additional_instruction="Use school/OHSAA/major local media sources."
    )

    # State championship win (critical)
    champ = evaluator.add_parallel(
        id="state_championship_win",
        desc="Won 2025 Ohio Division I state championship",
        parent=hs_node,
        critical=True
    )
    div_leaf = evaluator.add_leaf(
        id="division_classification",
        desc="Championship was in Division I (highest classification)",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim="The state championship was in Division I.",
        node=div_leaf,
        sources=_use_sources(hs.state_champ_urls),
        additional_instruction="Confirm division classification (Division I) on the OHSAA or reputable sources."
    )
    state_leaf = evaluator.add_leaf(
        id="state_identification",
        desc="Championship was in Ohio",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim="The state championship occurred in Ohio.",
        node=state_leaf,
        sources=_use_sources(hs.state_champ_urls),
        additional_instruction="Confirm the governing body (OHSAA) and Ohio context."
    )
    year_leaf = evaluator.add_leaf(
        id="championship_year",
        desc="Championship won in 2025 season",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim="The championship was won in the 2025 season.",
        node=year_leaf,
        sources=_use_sources(hs.state_champ_urls),
        additional_instruction="Confirm the year via title game recap or OHSAA records."
    )
    champ_ref_leaf = evaluator.add_leaf(
        id="state_champ_url_reference",
        desc="URL reference confirming state championship",
        parent=champ,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the Division I Ohio state championship in 2025 for the program.",
        node=champ_ref_leaf,
        sources=_use_sources(hs.state_champ_urls),
        additional_instruction="Use OHSAA and reputable news coverage."
    )

    # Historical first (critical core; non-critical extra separated)
    hist_core = evaluator.add_parallel(
        id="historical_first",
        desc="Championship was program's first in school history",
        parent=hs_node,
        critical=True
    )
    first_leaf = evaluator.add_leaf(
        id="first_title",
        desc="This was the school's first state championship",
        parent=hist_core,
        critical=True
    )
    await evaluator.verify(
        claim="This was the school's first-ever state championship in football.",
        node=first_leaf,
        sources=_use_sources(hs.historical_urls, hs.state_champ_urls),
        additional_instruction="Confirm 'first state title' language from reputable sources."
    )
    hist_ref_leaf = evaluator.add_leaf(
        id="historical_url_reference",
        desc="URL reference confirming historical first",
        parent=hist_core,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm this was the program's first state championship.",
        node=hist_ref_leaf,
        sources=_use_sources(hs.historical_urls),
        additional_instruction="Use OHSAA history pages or reputable local reports."
    )
    prev_extra = evaluator.add_parallel(
        id="previous_appearances_info",
        desc="Previous state championship appearances (non-critical)",
        parent=hs_node,
        critical=False
    )
    prev_leaf = evaluator.add_leaf(
        id="previous_appearances",
        desc="Confirm no prior state championships",
        parent=prev_extra,
        critical=False
    )
    await evaluator.verify(
        claim="The program had no prior state championships before 2025.",
        node=prev_leaf,
        sources=_use_sources(hs.historical_urls),
        additional_instruction="If a history page lists titles, ensure zero prior titles are indicated."
    )

    # Perfect season (critical core; non-critical score separated)
    perfect = evaluator.add_parallel(
        id="perfect_season",
        desc="Completed season with undefeated record",
        parent=hs_node,
        critical=True
    )
    record_leaf = evaluator.add_leaf(
        id="record_verification",
        desc="Program finished 15-0 or similar perfect record",
        parent=perfect,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program completed a perfect undefeated season ({_safe_text(hs.final_record, 'undefeated')}).",
        node=record_leaf,
        sources=_use_sources(hs.season_urls, hs.state_champ_urls),
        additional_instruction="Confirm final overall record showing zero losses."
    )
    opponent_city_leaf = evaluator.add_leaf(
        id="championship_game_opponent",
        desc="State championship opponent was from Cincinnati",
        parent=perfect,
        critical=True
    )
    await evaluator.verify(
        claim=f"The state championship opponent was a team from Cincinnati ({_safe_text(hs.championship_opponent_city, 'Cincinnati')}).",
        node=opponent_city_leaf,
        sources=_use_sources(hs.state_champ_urls, hs.season_urls),
        additional_instruction="Confirm opponent school and its Cincinnati location."
    )
    season_ref_leaf = evaluator.add_leaf(
        id="season_url_reference",
        desc="URL reference confirming perfect season details",
        parent=perfect,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the undefeated season and title game opponent/location details.",
        node=season_ref_leaf,
        sources=_use_sources(hs.season_urls, hs.state_champ_urls),
        additional_instruction="Use OHSAA brackets/recaps and reputable local news."
    )
    # Non-critical final score
    hs_extra = evaluator.add_parallel(
        id="hs_championship_game_details",
        desc="Championship game final score (non-critical)",
        parent=hs_node,
        critical=False
    )
    score_leaf = evaluator.add_leaf(
        id="championship_game_score",
        desc="Provide final score of championship game",
        parent=hs_extra,
        critical=False
    )
    await evaluator.verify(
        claim=f"The championship game final score was {_safe_text(hs.championship_final_score, 'stated')}.",
        node=score_leaf,
        sources=_use_sources(hs.state_champ_urls, hs.season_urls),
        additional_instruction="Confirm the final score on OHSAA or reputable media summaries."
    )


async def verify_college_basketball_coach(evaluator: Evaluator, parent_node, coach: Optional[CoachInfo]) -> None:
    coach_node = evaluator.add_parallel(
        id="college_basketball_coach",
        desc="Identify the college basketball coach who reached 900 career wins in 2021 with specific NCAA tournament credentials",
        parent=parent_node,
        critical=False
    )

    # Coach identification
    id_grp = evaluator.add_parallel(
        id="coach_identification",
        desc="Provide the name of the coach",
        parent=coach_node,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id="coach_name",
        desc="Full name of the coach",
        parent=id_grp,
        critical=True
    )
    await evaluator.verify(
        claim=f"The coach is '{_safe_text(coach.coach_name, 'the coach')}'.",
        node=name_leaf,
        sources=_use_sources(coach.identification_urls, coach.milestone_urls),
        additional_instruction="Confirm the coach's identity via official bios or reputable coverage."
    )
    id_ref_leaf = evaluator.add_leaf(
        id="identification_url_reference",
        desc="URL reference for coach identification",
        parent=id_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the coach's identity.",
        node=id_ref_leaf,
        sources=_use_sources(coach.identification_urls, coach.milestone_urls),
        additional_instruction="Use school athletics sites or long-standing media outlets."
    )
    # Non-critical current status
    status_grp = evaluator.add_parallel(
        id="coach_current_status_info",
        desc="Coach's current or most recent position (non-critical)",
        parent=coach_node,
        critical=False
    )
    status_leaf = evaluator.add_leaf(
        id="current_status",
        desc="Coach's current or most recent position",
        parent=status_grp,
        critical=False
    )
    await evaluator.verify(
        claim=f"The coach's current or most recent position is {_safe_text(coach.current_status, 'stated')}.",
        node=status_leaf,
        sources=_use_sources(coach.identification_urls),
        additional_instruction="Confirm the role/position if provided."
    )

    # Career milestone 900 wins in 2021 (critical)
    milestone_grp = evaluator.add_parallel(
        id="career_milestone",
        desc="Coach reached 900 career wins in 2021",
        parent=coach_node,
        critical=True
    )
    wins_leaf = evaluator.add_leaf(
        id="win_total",
        desc="Reached 900 career wins as head coach",
        parent=milestone_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The coach reached 900 career wins.",
        node=wins_leaf,
        sources=_use_sources(coach.milestone_urls),
        additional_instruction="Confirm coverage/bios listing the 900-win milestone."
    )
    year_leaf = evaluator.add_leaf(
        id="milestone_year",
        desc="Milestone achieved in 2021",
        parent=milestone_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The 900th win occurred in 2021.",
        node=year_leaf,
        sources=_use_sources(coach.milestone_urls),
        additional_instruction="Confirm the milestone date/year in reputable sources."
    )
    milestone_ref = evaluator.add_leaf(
        id="milestone_url_reference",
        desc="URL reference confirming 900-win milestone",
        parent=milestone_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the 900-win milestone and its timing.",
        node=milestone_ref,
        sources=_use_sources(coach.milestone_urls),
        additional_instruction="Use official school releases or credible news."
    )
    # Non-critical milestone game details
    milestone_extra = evaluator.add_parallel(
        id="milestone_game_details",
        desc="Details of 900th win (opponent/date) (non-critical)",
        parent=coach_node,
        critical=False
    )
    milestone_game_leaf = evaluator.add_leaf(
        id="milestone_game",
        desc="Provide details of 900th win (opponent, date)",
        parent=milestone_extra,
        critical=False
    )
    await evaluator.verify(
        claim=f"Details of the 900th win include: {_safe_text(coach.milestone_details, 'stated')}.",
        node=milestone_game_leaf,
        sources=_use_sources(coach.milestone_urls),
        additional_instruction="Confirm the opponent/date details if provided."
    )

    # Tournament appearances (critical)
    tour_grp = evaluator.add_parallel(
        id="tournament_appearances",
        desc="Coach made 25 NCAA tournament appearances",
        parent=coach_node,
        critical=True
    )
    count_leaf = evaluator.add_leaf(
        id="appearance_count",
        desc="25 NCAA tournament appearances as head coach",
        parent=tour_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The coach has made 25 NCAA tournament appearances.",
        node=count_leaf,
        sources=_use_sources(coach.tournament_urls, coach.identification_urls),
        additional_instruction="Confirm total NCAA tournament appearances count equals 25."
    )
    tour_ref_leaf = evaluator.add_leaf(
        id="tournament_url_reference",
        desc="URL reference confirming tournament appearances",
        parent=tour_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm the count of NCAA tournament appearances.",
        node=tour_ref_leaf,
        sources=_use_sources(coach.tournament_urls),
        additional_instruction="Use official records or reputable summaries."
    )
    # Non-critical tournament record
    tour_extra = evaluator.add_parallel(
        id="tournament_record_info",
        desc="NCAA tournament win-loss record (non-critical)",
        parent=coach_node,
        critical=False
    )
    tour_record_leaf = evaluator.add_leaf(
        id="tournament_record",
        desc="Provide NCAA tournament win-loss record",
        parent=tour_extra,
        critical=False
    )
    await evaluator.verify(
        claim=f"The coach's NCAA tournament record is {_safe_text(coach.tournament_record, 'stated')}.",
        node=tour_record_leaf,
        sources=_use_sources(coach.tournament_urls),
        additional_instruction="Confirm if a record value is provided."
    )

    # Final Four achievements (sequential)
    ff_seq = evaluator.add_sequential(
        id="final_four_achievements",
        desc="Led teams to Final Four at two different schools",
        parent=coach_node,
        critical=True
    )
    multi_schools = evaluator.add_parallel(
        id="multiple_schools",
        desc="Final Four appearances with at least two different programs",
        parent=ff_seq,
        critical=True
    )
    school_count_leaf = evaluator.add_leaf(
        id="school_count",
        desc="Made Final Four with two or more schools",
        parent=multi_schools,
        critical=True
    )
    await evaluator.verify(
        claim="The coach has made Final Four appearances at two different schools.",
        node=school_count_leaf,
        sources=_use_sources(coach.multiple_schools_urls, coach.years_urls),
        additional_instruction="Confirm at least two different schools are associated with Final Fours."
    )
    multi_ref_leaf = evaluator.add_leaf(
        id="multiple_schools_url_reference",
        desc="URL reference confirming multiple schools",
        parent=multi_schools,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm Final Four appearances at multiple schools.",
        node=multi_ref_leaf,
        sources=_use_sources(coach.multiple_schools_urls),
        additional_instruction="Use official or reputable coach bios/histories."
    )
    # Non-critical school names listed separately
    multi_extra = evaluator.add_parallel(
        id="final_four_school_names_info",
        desc="Identify the schools taken to Final Four (non-critical)",
        parent=coach_node,
        critical=False
    )
    school_names_leaf = evaluator.add_leaf(
        id="school_names",
        desc="Identify the schools taken to Final Four",
        parent=multi_extra,
        critical=False
    )
    await evaluator.verify(
        claim=f"The schools are: {_safe_text(coach.final_four_school_names, 'stated')}.",
        node=school_names_leaf,
        sources=_use_sources(coach.multiple_schools_urls),
        additional_instruction="Confirm the list of school names if provided."
    )

    # Specific years (parallel under sequential)
    years_grp = evaluator.add_parallel(
        id="specific_years",
        desc="Final Four appearances in 1992 and 2010",
        parent=ff_seq,
        critical=True
    )
    y1992_leaf = evaluator.add_leaf(
        id="year_1992",
        desc="Made Final Four in 1992",
        parent=years_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The coach made a Final Four in 1992.",
        node=y1992_leaf,
        sources=_use_sources(coach.years_urls, coach.multiple_schools_urls),
        additional_instruction="Confirm a Final Four berth in the year 1992."
    )
    y2010_leaf = evaluator.add_leaf(
        id="year_2010",
        desc="Made Final Four in 2010",
        parent=years_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The coach made a Final Four in 2010.",
        node=y2010_leaf,
        sources=_use_sources(coach.years_urls, coach.multiple_schools_urls),
        additional_instruction="Confirm a Final Four berth in the year 2010."
    )
    years_ref_leaf = evaluator.add_leaf(
        id="years_url_reference",
        desc="URL reference confirming specific Final Four years",
        parent=years_grp,
        critical=True
    )
    await evaluator.verify(
        claim="The sources confirm Final Four appearances in 1992 and 2010.",
        node=years_ref_leaf,
        sources=_use_sources(coach.years_urls),
        additional_instruction="Use official NCAA records or reputable bios."
    )
    # Non-critical year-school mapping
    years_extra = evaluator.add_parallel(
        id="final_four_year_school_mapping",
        desc="Identify schools for 1992 and 2010 Final Fours (non-critical)",
        parent=coach_node,
        critical=False
    )
    y1992_school_leaf = evaluator.add_leaf(
        id="year_1992_school",
        desc="Identify school for 1992 Final Four",
        parent=years_extra,
        critical=False
    )
    await evaluator.verify(
        claim="The school associated with the 1992 Final Four appearance is correctly identified.",
        node=y1992_school_leaf,
        sources=_use_sources(coach.years_urls, coach.multiple_schools_urls),
        additional_instruction="Confirm which school the coach led in 1992."
    )
    y2010_school_leaf = evaluator.add_leaf(
        id="year_2010_school",
        desc="Identify school for 2010 Final Four",
        parent=years_extra,
        critical=False
    )
    await evaluator.verify(
        claim="The school associated with the 2010 Final Four appearance is correctly identified.",
        node=y2010_school_leaf,
        sources=_use_sources(coach.years_urls, coach.multiple_schools_urls),
        additional_instruction="Confirm which school the coach led in 2010."
    )


# ---------------------------- Main Entrypoint ------------------------------ #

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

    # Extract all entities and references
    extracted: AllExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllExtraction,
        extraction_name="extracted_entities"
    )

    # Build verification subtrees for each of the four items
    await verify_fcs_program(evaluator, root, extracted.fcs or FCSInfo())
    await verify_ivy_program(evaluator, root, extracted.ivy or IvyInfo())
    await verify_high_school_program(evaluator, root, extracted.hs or HSInfo())
    await verify_college_basketball_coach(evaluator, root, extracted.coach or CoachInfo())

    # Return evaluator summary
    return evaluator.get_summary()