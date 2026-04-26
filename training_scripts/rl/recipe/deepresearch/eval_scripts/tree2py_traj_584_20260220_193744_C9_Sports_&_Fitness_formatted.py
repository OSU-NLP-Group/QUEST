import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "super_bowl_lx_key_players_analysis"
TASK_DESCRIPTION = """
Provide comprehensive information about the key players from Super Bowl LX (played on February 8, 2026) who meet the following criteria, including their college backgrounds, regular season performance, and playoff statistics:

1. Super Bowl MVP (Running Back): Identify the running back who won Super Bowl LX MVP. Provide:
   - Confirmation of MVP award
   - Super Bowl rushing statistics (rushing yards ≥135, rushing attempts)
   - Super Bowl receiving statistics (receiving yards, number of receptions)
   - College university where they played
   - Their college season year for statistical reference
   - College rushing statistics from that season (rushing yards ≥1,600, rushing touchdowns)

2. Defensive Standout (Cornerback): Identify a defensive player from the winning team who recorded at least 1 sack in Super Bowl LX. Provide:
   - Confirmation they played for the winning team
   - Super Bowl defensive statistics (solo tackles, sacks ≥1, forced fumbles, pass deflections)
   - College university where they played
   - Their college season year for statistical reference
   - College defensive statistics from that season (tackles ≥50, pass breakups)

3. Winning Team Quarterback: Identify the starting quarterback for the Super Bowl LX winning team. Provide:
   - Confirmation of their QB role and team
   - Their team's complete playoff path including:
     * 2025 regular season record and conference seed position
     * Wild card round status
     * Divisional round opponent and score
     * Conference championship opponent, score, and date

4. Losing Team Quarterback: Identify the starting quarterback for the Super Bowl LX losing team. Provide:
   - Confirmation of their QB role and team
   - 2025 regular season statistics (passing yards ≥4,000, passing touchdowns ≥30, interceptions)
   - College university where they played
   - Their college season year for statistical reference
   - College passing statistics from that season (passing yards ≥3,500, passing touchdowns)
   - Their team's playoff path including:
     * 2025 regular season record
     * Conference championship opponent, score, and date

5. Game Details: Provide:
   - Super Bowl LX date
   - Venue name and location
   - Final score
   - Stadium seating capacity

For each piece of information, include supporting reference URLs from your research.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
def _default_list() -> List[str]:
    return []

class MVPInfo(BaseModel):
    name: Optional[str] = None
    team: Optional[str] = None
    # Award confirmation
    award_sources: List[str] = Field(default_factory=_default_list)
    # Super Bowl rushing
    sb_rushing_yards: Optional[str] = None
    sb_rushing_yards_sources: List[str] = Field(default_factory=_default_list)
    sb_rushing_attempts: Optional[str] = None
    sb_rushing_attempts_sources: List[str] = Field(default_factory=_default_list)
    # Super Bowl receiving
    sb_receiving_yards: Optional[str] = None
    sb_receiving_yards_sources: List[str] = Field(default_factory=_default_list)
    sb_receptions: Optional[str] = None
    sb_receptions_sources: List[str] = Field(default_factory=_default_list)
    # College background
    college_university: Optional[str] = None
    college_university_sources: List[str] = Field(default_factory=_default_list)
    college_season_year: Optional[str] = None
    college_season_year_sources: List[str] = Field(default_factory=_default_list)
    college_rushing_yards: Optional[str] = None
    college_rushing_yards_sources: List[str] = Field(default_factory=_default_list)
    college_rushing_tds: Optional[str] = None
    college_rushing_tds_sources: List[str] = Field(default_factory=_default_list)

class DefensiveInfo(BaseModel):
    name: Optional[str] = None
    position: Optional[str] = None  # e.g., Cornerback
    team: Optional[str] = None
    # Winning team confirmation
    winning_team_confirmation_sources: List[str] = Field(default_factory=_default_list)
    # Super Bowl defensive stats
    sb_solo_tackles: Optional[str] = None
    sb_solo_tackles_sources: List[str] = Field(default_factory=_default_list)
    sb_sacks: Optional[str] = None
    sb_sacks_sources: List[str] = Field(default_factory=_default_list)
    sb_forced_fumbles: Optional[str] = None
    sb_forced_fumbles_sources: List[str] = Field(default_factory=_default_list)
    sb_pass_deflections: Optional[str] = None
    sb_pass_deflections_sources: List[str] = Field(default_factory=_default_list)
    # College background
    college_university: Optional[str] = None
    college_university_sources: List[str] = Field(default_factory=_default_list)
    college_season_year: Optional[str] = None
    college_season_year_sources: List[str] = Field(default_factory=_default_list)
    college_tackles: Optional[str] = None
    college_tackles_sources: List[str] = Field(default_factory=_default_list)
    college_pass_breakups: Optional[str] = None
    college_pass_breakups_sources: List[str] = Field(default_factory=_default_list)

class WinningQBInfo(BaseModel):
    name: Optional[str] = None
    team: Optional[str] = None
    role_confirmation_sources: List[str] = Field(default_factory=_default_list)
    # Playoff path
    season_record_and_seed: Optional[str] = None
    season_record_sources: List[str] = Field(default_factory=_default_list)
    wild_card_status: Optional[str] = None
    wild_card_sources: List[str] = Field(default_factory=_default_list)
    divisional_opponent_and_score: Optional[str] = None
    divisional_sources: List[str] = Field(default_factory=_default_list)
    conf_champ_opponent_score_date: Optional[str] = None
    conf_champ_sources: List[str] = Field(default_factory=_default_list)

class LosingQBInfo(BaseModel):
    name: Optional[str] = None
    team: Optional[str] = None
    role_confirmation_sources: List[str] = Field(default_factory=_default_list)
    # Regular season performance
    reg_passing_yards: Optional[str] = None
    reg_passing_yards_sources: List[str] = Field(default_factory=_default_list)
    reg_passing_tds: Optional[str] = None
    reg_passing_tds_sources: List[str] = Field(default_factory=_default_list)
    reg_interceptions: Optional[str] = None
    reg_interceptions_sources: List[str] = Field(default_factory=_default_list)
    # College background
    college_university: Optional[str] = None
    college_university_sources: List[str] = Field(default_factory=_default_list)
    college_season_year: Optional[str] = None
    college_season_year_sources: List[str] = Field(default_factory=_default_list)
    college_passing_yards: Optional[str] = None
    college_passing_yards_sources: List[str] = Field(default_factory=_default_list)
    college_passing_tds: Optional[str] = None
    college_passing_tds_sources: List[str] = Field(default_factory=_default_list)
    # Playoff path
    losing_team_regular_season_record: Optional[str] = None
    losing_team_regular_season_sources: List[str] = Field(default_factory=_default_list)
    conf_champ_opponent_score_date: Optional[str] = None
    conf_champ_sources: List[str] = Field(default_factory=_default_list)

class GameDetails(BaseModel):
    date: Optional[str] = None
    date_sources: List[str] = Field(default_factory=_default_list)
    venue_name_and_location: Optional[str] = None
    venue_sources: List[str] = Field(default_factory=_default_list)
    final_score: Optional[str] = None
    final_score_sources: List[str] = Field(default_factory=_default_list)
    stadium_capacity: Optional[str] = None
    stadium_capacity_sources: List[str] = Field(default_factory=_default_list)

class SBKeyPlayersExtraction(BaseModel):
    mvp_rb: Optional[MVPInfo] = None
    defensive_cb: Optional[DefensiveInfo] = None
    winning_qb: Optional[WinningQBInfo] = None
    losing_qb: Optional[LosingQBInfo] = None
    game: Optional[GameDetails] = None

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_sb_key_players() -> str:
    return """
    Extract structured information from the answer about Super Bowl LX (February 8, 2026).
    Return a JSON object with keys: mvp_rb, defensive_cb, winning_qb, losing_qb, game.
    For any missing item or value, return null (for strings) or an empty list (for URLs).

    For mvp_rb (Super Bowl MVP Running Back), extract:
      - name
      - team
      - award_sources: list of URLs confirming MVP award
      - sb_rushing_yards, sb_rushing_yards_sources: list of URLs
      - sb_rushing_attempts, sb_rushing_attempts_sources: list of URLs
      - sb_receiving_yards, sb_receiving_yards_sources: list of URLs
      - sb_receptions, sb_receptions_sources: list of URLs
      - college_university, college_university_sources: list of URLs
      - college_season_year, college_season_year_sources: list of URLs
      - college_rushing_yards, college_rushing_yards_sources: list of URLs
      - college_rushing_tds, college_rushing_tds_sources: list of URLs

    For defensive_cb (Defensive standout Cornerback from winning team), extract:
      - name
      - position
      - team
      - winning_team_confirmation_sources: list of URLs confirming the player was on the winning team
      - sb_solo_tackles, sb_solo_tackles_sources: list of URLs
      - sb_sacks, sb_sacks_sources: list of URLs
      - sb_forced_fumbles, sb_forced_fumbles_sources: list of URLs
      - sb_pass_deflections, sb_pass_deflections_sources: list of URLs
      - college_university, college_university_sources: list of URLs
      - college_season_year, college_season_year_sources: list of URLs
      - college_tackles, college_tackles_sources: list of URLs
      - college_pass_breakups, college_pass_breakups_sources: list of URLs

    For winning_qb (Winning team starting QB), extract:
      - name
      - team
      - role_confirmation_sources: list of URLs confirming QB role and winning team
      - season_record_and_seed, season_record_sources: list of URLs
      - wild_card_status, wild_card_sources: list of URLs
      - divisional_opponent_and_score, divisional_sources: list of URLs
      - conf_champ_opponent_score_date, conf_champ_sources: list of URLs

    For losing_qb (Losing team starting QB), extract:
      - name
      - team
      - role_confirmation_sources: list of URLs confirming QB role and losing team
      - reg_passing_yards, reg_passing_yards_sources: list of URLs
      - reg_passing_tds, reg_passing_tds_sources: list of URLs
      - reg_interceptions, reg_interceptions_sources: list of URLs
      - college_university, college_university_sources: list of URLs
      - college_season_year, college_season_year_sources: list of URLs
      - college_passing_yards, college_passing_yards_sources: list of URLs
      - college_passing_tds, college_passing_tds_sources: list of URLs
      - losing_team_regular_season_record, losing_team_regular_season_sources: list of URLs
      - conf_champ_opponent_score_date, conf_champ_sources: list of URLs

    For game (Super Bowl LX game details), extract:
      - date, date_sources: list of URLs
      - venue_name_and_location, venue_sources: list of URLs
      - final_score, final_score_sources: list of URLs
      - stadium_capacity, stadium_capacity_sources: list of URLs

    URL extraction rules:
      - Extract only actual URLs mentioned in the answer (including markdown links); if none, return an empty list.
      - Do not invent URLs. If missing protocol, prepend http://.
    """

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]

def safe_text(val: Optional[str], fallback: str = "") -> str:
    return val.strip() if isinstance(val, str) else fallback

def add_url_existence_leaf(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    urls: Optional[List[str]],
    critical: bool = True
):
    return evaluator.add_custom_node(
        result=len(safe_sources(urls)) > 0,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_mvp_rb(evaluator: Evaluator, parent_node, info: Optional[MVPInfo]) -> None:
    mvp_node = evaluator.add_parallel(
        id="Super_Bowl_MVP_Running_Back",
        desc="Information about the Super Bowl LX MVP who was a running back from the winning team",
        parent=parent_node,
        critical=False,
    )
    if not info:
        # If no MVP info provided, still create minimal leaves to fail explicitly
        add_url_existence_leaf(
            evaluator, mvp_node, "MVP_Award_URL",
            "Valid reference URL confirming MVP award", urls=[], critical=True
        )
        return

    # MVP Award Confirmation group
    award_group = evaluator.add_parallel(
        id="MVP_Award_Confirmation_Group",
        desc="Confirms the identified player was officially named Super Bowl LX MVP",
        parent=mvp_node,
        critical=False
    )
    # Claim leaf
    mvp_claim_leaf = evaluator.add_leaf(
        id="MVP_Award_Confirmation",
        desc="Confirms the identified player was officially named Super Bowl LX MVP",
        parent=award_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{safe_text(info.name, 'The MVP')} was named Super Bowl LX MVP.",
        node=mvp_claim_leaf,
        sources=safe_sources(info.award_sources),
        additional_instruction="Verify on official or reputable sources (e.g., NFL, AP, team, major outlets) that the named running back won Super Bowl LX MVP."
    )
    # URL existence leaf
    add_url_existence_leaf(
        evaluator, award_group, "MVP_Award_URL",
        "Valid reference URL confirming MVP award",
        urls=info.award_sources,
        critical=True
    )

    # Super Bowl rushing performance (critical)
    rush_group = evaluator.add_parallel(
        id="MVP_Super_Bowl_Rushing_Performance",
        desc="Super Bowl LX rushing statistics for the MVP",
        parent=mvp_node,
        critical=False
    )
    # Rushing yards (threshold ≥135)
    rush_yds_leaf = evaluator.add_leaf(
        id="Rushing_Yards",
        desc="Rushing yards in Super Bowl LX (must be ≥135 yards)",
        parent=rush_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"In Super Bowl LX, {safe_text(info.name, 'the MVP')} rushed for at least 135 yards.",
        node=rush_yds_leaf,
        sources=safe_sources(info.sb_rushing_yards_sources),
        additional_instruction="Confirm the player's Super Bowl LX rushing yards are ≥135; small rounding differences are acceptable if clearly ≥135."
    )
    add_url_existence_leaf(
        evaluator, rush_group, "Rushing_Yards_URL",
        "Reference URL for rushing yards statistic",
        urls=info.sb_rushing_yards_sources,
        critical=True
    )

    # Rushing attempts
    rush_att_leaf = evaluator.add_leaf(
        id="Rushing_Attempts",
        desc="Number of rushing attempts in Super Bowl LX",
        parent=rush_group,
        critical=True
    )
    attempts_txt = safe_text(info.sb_rushing_attempts, "")
    claim_attempts = f"In Super Bowl LX, {safe_text(info.name, 'the MVP')} had {attempts_txt} rushing attempts." if attempts_txt else f"In Super Bowl LX, {safe_text(info.name, 'the MVP')} had a specific number of rushing attempts."
    await evaluator.verify(
        claim=claim_attempts,
        node=rush_att_leaf,
        sources=safe_sources(info.sb_rushing_attempts_sources),
        additional_instruction="Verify the exact number of rushing attempts for the player in Super Bowl LX."
    )
    add_url_existence_leaf(
        evaluator, rush_group, "Rushing_Attempts_URL",
        "Reference URL for rushing attempts statistic",
        urls=info.sb_rushing_attempts_sources,
        critical=True
    )

    # Super Bowl receiving performance (non-critical)
    recv_group = evaluator.add_parallel(
        id="MVP_Super_Bowl_Receiving_Performance",
        desc="Super Bowl LX receiving statistics for the MVP",
        parent=mvp_node,
        critical=False
    )
    recv_yds_leaf = evaluator.add_leaf(
        id="Receiving_Yards",
        desc="Receiving yards in Super Bowl LX",
        parent=recv_group,
        critical=False
    )
    ry_txt = safe_text(info.sb_receiving_yards, "")
    claim_ry = f"In Super Bowl LX, {safe_text(info.name, 'the MVP')} had {ry_txt} receiving yards." if ry_txt else f"In Super Bowl LX, {safe_text(info.name, 'the MVP')} recorded receiving yards."
    await evaluator.verify(
        claim=claim_ry,
        node=recv_yds_leaf,
        sources=safe_sources(info.sb_receiving_yards_sources),
        additional_instruction="Verify the player's Super Bowl LX receiving yards; allow minor formatting variations."
    )
    add_url_existence_leaf(
        evaluator, recv_group, "Receiving_Yards_URL",
        "Reference URL for receiving yards statistic",
        urls=info.sb_receiving_yards_sources,
        critical=True
    )

    receptions_leaf = evaluator.add_leaf(
        id="Receptions",
        desc="Number of receptions in Super Bowl LX",
        parent=recv_group,
        critical=False
    )
    rec_txt = safe_text(info.sb_receptions, "")
    claim_rec = f"In Super Bowl LX, {safe_text(info.name, 'the MVP')} made {rec_txt} receptions." if rec_txt else f"In Super Bowl LX, {safe_text(info.name, 'the MVP')} made a certain number of receptions."
    await evaluator.verify(
        claim=claim_rec,
        node=receptions_leaf,
        sources=safe_sources(info.sb_receptions_sources),
        additional_instruction="Verify the number of receptions for the player in Super Bowl LX."
    )
    add_url_existence_leaf(
        evaluator, recv_group, "Receptions_URL",
        "Reference URL for receptions statistic",
        urls=info.sb_receptions_sources,
        critical=True
    )

    # College background (mixed criticality handled at leaf level)
    college_group = evaluator.add_parallel(
        id="MVP_College_Background",
        desc="College football information for the Super Bowl MVP",
        parent=mvp_node,
        critical=False
    )
    # College university
    college_univ_leaf = evaluator.add_leaf(
        id="College_University",
        desc="University where the MVP played college football",
        parent=college_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{safe_text(info.name, 'The MVP')} played college football at {safe_text(info.college_university, 'the stated university')}.",
        node=college_univ_leaf,
        sources=safe_sources(info.college_university_sources),
        additional_instruction="Verify the player's college affiliation using reliable sources (school bio, NCAA, reputable profiles)."
    )
    add_url_existence_leaf(
        evaluator, college_group, "College_University_URL",
        "Reference URL confirming college affiliation",
        urls=info.college_university_sources,
        critical=True
    )

    # College season year
    college_year_leaf = evaluator.add_leaf(
        id="College_Season_Year",
        desc="Specific college season year for statistical reference",
        parent=college_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The relevant college season year for {safe_text(info.name, 'the player')} is {safe_text(info.college_season_year, 'the stated year')}.",
        node=college_year_leaf,
        sources=safe_sources(info.college_season_year_sources),
        additional_instruction="Verify the season year referenced for the player's college statistics."
    )
    add_url_existence_leaf(
        evaluator, college_group, "College_Season_Year_URL",
        "Reference URL confirming the season year",
        urls=info.college_season_year_sources,
        critical=True
    )

    # College rushing yards threshold ≥1600
    college_yds_leaf = evaluator.add_leaf(
        id="College_Rushing_Yards",
        desc="Rushing yards in the specified college season (must be ≥1,600 yards)",
        parent=college_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {safe_text(info.college_season_year, 'the stated year')}, {safe_text(info.name, 'the player')} rushed for at least 1,600 yards in college.",
        node=college_yds_leaf,
        sources=safe_sources(info.college_rushing_yards_sources),
        additional_instruction="Confirm the season rushing yards are ≥1,600. Allow minor rounding if clearly above threshold."
    )
    add_url_existence_leaf(
        evaluator, college_group, "College_Rushing_Yards_URL",
        "Reference URL for college rushing yards",
        urls=info.college_rushing_yards_sources,
        critical=True
    )

    # College rushing touchdowns (non-critical)
    college_tds_leaf = evaluator.add_leaf(
        id="College_Rushing_Touchdowns",
        desc="Rushing touchdowns in the specified college season",
        parent=college_group,
        critical=False
    )
    td_txt = safe_text(info.college_rushing_tds, "")
    claim_tds = f"In {safe_text(info.college_season_year, 'the stated year')}, {safe_text(info.name, 'the player')} recorded {td_txt} rushing touchdowns." if td_txt else f"In {safe_text(info.college_season_year, 'the stated year')}, {safe_text(info.name, 'the player')} recorded rushing touchdowns."
    await evaluator.verify(
        claim=claim_tds,
        node=college_tds_leaf,
        sources=safe_sources(info.college_rushing_tds_sources),
        additional_instruction="Verify the player's rushing touchdowns in that college season."
    )
    add_url_existence_leaf(
        evaluator, college_group, "College_Rushing_Touchdowns_URL",
        "Reference URL for college rushing touchdowns",
        urls=info.college_rushing_tds_sources,
        critical=True
    )

async def verify_defensive_cb(evaluator: Evaluator, parent_node, info: Optional[DefensiveInfo]) -> None:
    def_node = evaluator.add_parallel(
        id="Defensive_Standout_Cornerback",
        desc="Information about a defensive standout from the winning team who recorded at least one sack in Super Bowl LX",
        parent=parent_node,
        critical=False,
    )
    if not info:
        add_url_existence_leaf(
            evaluator, def_node, "Defensive_Team_URL",
            "Reference URL confirming team affiliation", urls=[], critical=True
        )
        return

    # Team confirmation
    team_group = evaluator.add_parallel(
        id="Defensive_Player_Team_Confirmation_Group",
        desc="Confirms the player was on the Super Bowl LX winning team",
        parent=def_node,
        critical=False
    )
    team_leaf = evaluator.add_leaf(
        id="Defensive_Player_Team_Confirmation",
        desc="Confirms the player was on the Super Bowl LX winning team",
        parent=team_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{safe_text(info.name, 'The defensive player')} played for {safe_text(info.team, 'the stated team')}, which won Super Bowl LX.",
        node=team_leaf,
        sources=safe_sources(info.winning_team_confirmation_sources),
        additional_instruction="Verify that the player's team is the Super Bowl LX winning team and the player was on that roster."
    )
    add_url_existence_leaf(
        evaluator, team_group, "Defensive_Team_URL",
        "Reference URL confirming team affiliation",
        urls=info.winning_team_confirmation_sources,
        critical=True
    )

    # Super Bowl defensive performance
    perf_group = evaluator.add_parallel(
        id="Defensive_Super_Bowl_Performance",
        desc="Super Bowl LX defensive statistics",
        parent=def_node,
        critical=False
    )
    # Solo tackles (non-critical)
    solo_leaf = evaluator.add_leaf(
        id="Solo_Tackles",
        desc="Number of solo tackles in Super Bowl LX",
        parent=perf_group,
        critical=False
    )
    solo_txt = safe_text(info.sb_solo_tackles, "")
    claim_solo = f"In Super Bowl LX, {safe_text(info.name, 'the defensive player')} had {solo_txt} solo tackles." if solo_txt else f"In Super Bowl LX, {safe_text(info.name, 'the defensive player')} recorded solo tackles."
    await evaluator.verify(
        claim=claim_solo,
        node=solo_leaf,
        sources=safe_sources(info.sb_solo_tackles_sources),
        additional_instruction="Verify the player's solo tackles total for Super Bowl LX."
    )
    add_url_existence_leaf(
        evaluator, perf_group, "Solo_Tackles_URL",
        "Reference URL for solo tackles statistic",
        urls=info.sb_solo_tackles_sources,
        critical=True
    )

    # Sacks (critical, threshold ≥1)
    sacks_leaf = evaluator.add_leaf(
        id="Sacks",
        desc="Number of sacks in Super Bowl LX (must be ≥1)",
        parent=perf_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"In Super Bowl LX, {safe_text(info.name, 'the defensive player')} recorded at least 1 sack.",
        node=sacks_leaf,
        sources=safe_sources(info.sb_sacks_sources),
        additional_instruction="Confirm the player recorded ≥1 sack in Super Bowl LX."
    )
    add_url_existence_leaf(
        evaluator, perf_group, "Sacks_URL",
        "Reference URL for sacks statistic",
        urls=info.sb_sacks_sources,
        critical=True
    )

    # Forced fumbles (non-critical)
    ff_leaf = evaluator.add_leaf(
        id="Forced_Fumbles",
        desc="Number of forced fumbles in Super Bowl LX",
        parent=perf_group,
        critical=False
    )
    ff_txt = safe_text(info.sb_forced_fumbles, "")
    claim_ff = f"In Super Bowl LX, {safe_text(info.name, 'the defensive player')} had {ff_txt} forced fumbles." if ff_txt else f"In Super Bowl LX, {safe_text(info.name, 'the defensive player')} had forced fumbles."
    await evaluator.verify(
        claim=claim_ff,
        node=ff_leaf,
        sources=safe_sources(info.sb_forced_fumbles_sources),
        additional_instruction="Verify the number of forced fumbles for the player in Super Bowl LX."
    )
    add_url_existence_leaf(
        evaluator, perf_group, "Forced_Fumbles_URL",
        "Reference URL for forced fumbles statistic",
        urls=info.sb_forced_fumbles_sources,
        critical=True
    )

    # Pass deflections (non-critical)
    pd_leaf = evaluator.add_leaf(
        id="Pass_Deflections",
        desc="Number of pass deflections in Super Bowl LX",
        parent=perf_group,
        critical=False
    )
    pd_txt = safe_text(info.sb_pass_deflections, "")
    claim_pd = f"In Super Bowl LX, {safe_text(info.name, 'the defensive player')} had {pd_txt} pass deflections." if pd_txt else f"In Super Bowl LX, {safe_text(info.name, 'the defensive player')} had pass deflections."
    await evaluator.verify(
        claim=claim_pd,
        node=pd_leaf,
        sources=safe_sources(info.sb_pass_deflections_sources),
        additional_instruction="Verify the number of pass deflections (passes defended/PD) for the player in Super Bowl LX."
    )
    add_url_existence_leaf(
        evaluator, perf_group, "Pass_Deflections_URL",
        "Reference URL for pass deflections statistic",
        urls=info.sb_pass_deflections_sources,
        critical=True
    )

    # College background
    def_college_group = evaluator.add_parallel(
        id="Defensive_College_Background",
        desc="College football information for the defensive standout",
        parent=def_node,
        critical=False
    )
    # College university (critical)
    def_col_univ_leaf = evaluator.add_leaf(
        id="Defensive_College_University",
        desc="University where the defensive player played college football",
        parent=def_college_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{safe_text(info.name, 'The defensive player')} played college football at {safe_text(info.college_university, 'the stated university')}.",
        node=def_col_univ_leaf,
        sources=safe_sources(info.college_university_sources),
        additional_instruction="Verify college affiliation using reliable sources."
    )
    add_url_existence_leaf(
        evaluator, def_college_group, "Defensive_College_University_URL",
        "Reference URL confirming college affiliation",
        urls=info.college_university_sources,
        critical=True
    )

    # College season year (critical)
    def_col_year_leaf = evaluator.add_leaf(
        id="Defensive_College_Season_Year",
        desc="Specific college season year for statistical reference",
        parent=def_college_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The relevant college season year for {safe_text(info.name, 'the defensive player')} is {safe_text(info.college_season_year, 'the stated year')}.",
        node=def_col_year_leaf,
        sources=safe_sources(info.college_season_year_sources),
        additional_instruction="Verify the season year referenced for the player's college statistics."
    )
    add_url_existence_leaf(
        evaluator, def_college_group, "Defensive_College_Season_Year_URL",
        "Reference URL confirming the season year",
        urls=info.college_season_year_sources,
        critical=True
    )

    # College tackles threshold ≥50 (critical)
    def_col_tackles_leaf = evaluator.add_leaf(
        id="Defensive_College_Tackles",
        desc="Total tackles in the specified college season (must be ≥50)",
        parent=def_college_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {safe_text(info.college_season_year, 'the stated year')}, {safe_text(info.name, 'the defensive player')} recorded at least 50 total tackles in college.",
        node=def_col_tackles_leaf,
        sources=safe_sources(info.college_tackles_sources),
        additional_instruction="Confirm the player's college tackles are ≥50 for that season."
    )
    add_url_existence_leaf(
        evaluator, def_college_group, "Defensive_College_Tackles_URL",
        "Reference URL for college tackles",
        urls=info.college_tackles_sources,
        critical=True
    )

    # College pass breakups (non-critical)
    def_col_pbu_leaf = evaluator.add_leaf(
        id="Defensive_College_Pass_Breakups",
        desc="Pass breakups in the specified college season",
        parent=def_college_group,
        critical=False
    )
    pbu_txt = safe_text(info.college_pass_breakups, "")
    claim_pbu = f"In {safe_text(info.college_season_year, 'the stated year')}, {safe_text(info.name, 'the defensive player')} recorded {pbu_txt} pass breakups." if pbu_txt else f"In {safe_text(info.college_season_year, 'the stated year')}, {safe_text(info.name, 'the defensive player')} recorded pass breakups."
    await evaluator.verify(
        claim=claim_pbu,
        node=def_col_pbu_leaf,
        sources=safe_sources(info.college_pass_breakups_sources),
        additional_instruction="Verify the pass breakups (PBUs) for the player in that college season."
    )
    add_url_existence_leaf(
        evaluator, def_college_group, "Defensive_College_Pass_Breakups_URL",
        "Reference URL for college pass breakups",
        urls=info.college_pass_breakups_sources,
        critical=True
    )

async def verify_winning_qb(evaluator: Evaluator, parent_node, info: Optional[WinningQBInfo]) -> None:
    win_qb_node = evaluator.add_parallel(
        id="Winning_Team_Quarterback",
        desc="Information about the starting quarterback for the Super Bowl LX winning team",
        parent=parent_node,
        critical=False,
    )
    if not info:
        add_url_existence_leaf(
            evaluator, win_qb_node, "Winning_QB_Team_URL",
            "Reference URL confirming QB role and team", urls=[], critical=True
        )
        return

    # Team/QB role confirmation
    confirm_group = evaluator.add_parallel(
        id="Winning_QB_Team_Confirmation_Group",
        desc="Confirms the quarterback was the starting QB for the winning team in Super Bowl LX",
        parent=win_qb_node,
        critical=False
    )
    confirm_leaf = evaluator.add_leaf(
        id="Winning_QB_Team_Confirmation",
        desc="Confirms the quarterback was the starting QB for the winning team in Super Bowl LX",
        parent=confirm_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{safe_text(info.name, 'The quarterback')} was the starting QB for {safe_text(info.team, 'the stated team')}, which won Super Bowl LX.",
        node=confirm_leaf,
        sources=safe_sources(info.role_confirmation_sources),
        additional_instruction="Verify the QB's starting role for the Super Bowl LX winning team."
    )
    add_url_existence_leaf(
        evaluator, confirm_group, "Winning_QB_Team_URL",
        "Reference URL confirming QB role and team",
        urls=info.role_confirmation_sources,
        critical=True
    )

    # Playoff path (sequential)
    path_seq = evaluator.add_sequential(
        id="Winning_QB_Playoff_Path",
        desc="Winning team's playoff path to Super Bowl LX",
        parent=win_qb_node,
        critical=False
    )
    # Regular season record & seed
    add_url_existence_leaf(
        evaluator, path_seq, "Regular_Season_URL",
        "Reference URL for regular season record",
        urls=info.season_record_sources,
        critical=True
    )
    reg_leaf = evaluator.add_leaf(
        id="Regular_Season_Record",
        desc="Winning team's 2025 regular season record and conference seed position",
        parent=path_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2025, the {safe_text(info.team, 'winning team')} had {safe_text(info.season_record_and_seed, 'the stated record/seed')}.",
        node=reg_leaf,
        sources=safe_sources(info.season_record_sources),
        additional_instruction="Verify both the regular season record and conference seed for the winning team."
    )

    # Wild card round status (include but treat non-critical in sequence by still verifying)
    add_url_existence_leaf(
        evaluator, path_seq, "Wild_Card_URL",
        "Reference URL confirming wild card round status",
        urls=info.wild_card_sources,
        critical=True
    )
    wild_leaf = evaluator.add_leaf(
        id="Wild_Card_Round",
        desc="Wild card round status",
        parent=path_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Wild card round status for the {safe_text(info.team, 'winning team')} was: {safe_text(info.wild_card_status, 'the stated status')}.",
        node=wild_leaf,
        sources=safe_sources(info.wild_card_sources),
        additional_instruction="Verify whether the team had a bye, did not play, or the specific wild card game status."
    )

    # Divisional round opponent and score
    add_url_existence_leaf(
        evaluator, path_seq, "Divisional_URL",
        "Reference URL for divisional round game",
        urls=info.divisional_sources,
        critical=True
    )
    div_leaf = evaluator.add_leaf(
        id="Divisional_Round_Result",
        desc="Divisional round opponent and score",
        parent=path_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the Divisional Round, the {safe_text(info.team, 'winning team')} faced {safe_text(info.divisional_opponent_and_score, 'the stated opponent/score')}.",
        node=div_leaf,
        sources=safe_sources(info.divisional_sources),
        additional_instruction="Verify the divisional round opponent and final score."
    )

    # Conference championship opponent, score, date
    add_url_existence_leaf(
        evaluator, path_seq, "Conference_Championship_URL",
        "Reference URL for conference championship game",
        urls=info.conf_champ_sources,
        critical=True
    )
    cc_leaf = evaluator.add_leaf(
        id="Conference_Championship_Result",
        desc="Conference championship opponent, score, and date",
        parent=path_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the Conference Championship, the {safe_text(info.team, 'winning team')} played {safe_text(info.conf_champ_opponent_score_date, 'the stated opponent/score/date')}.",
        node=cc_leaf,
        sources=safe_sources(info.conf_champ_sources),
        additional_instruction="Verify the opponent, score, and date of the conference championship game."
    )

async def verify_losing_qb(evaluator: Evaluator, parent_node, info: Optional[LosingQBInfo]) -> None:
    lose_qb_node = evaluator.add_parallel(
        id="Losing_Team_Quarterback",
        desc="Information about the starting quarterback for the Super Bowl LX losing team",
        parent=parent_node,
        critical=False,
    )
    if not info:
        add_url_existence_leaf(
            evaluator, lose_qb_node, "Losing_QB_Team_URL",
            "Reference URL confirming QB role and team", urls=[], critical=True
        )
        return

    # Team/QB role confirmation
    lose_confirm_group = evaluator.add_parallel(
        id="Losing_QB_Team_Confirmation_Group",
        desc="Confirms the quarterback was the starting QB for the losing team in Super Bowl LX",
        parent=lose_qb_node,
        critical=False
    )
    lose_confirm_leaf = evaluator.add_leaf(
        id="Losing_QB_Team_Confirmation",
        desc="Confirms the quarterback was the starting QB for the losing team in Super Bowl LX",
        parent=lose_confirm_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{safe_text(info.name, 'The quarterback')} was the starting QB for {safe_text(info.team, 'the stated team')}, which lost Super Bowl LX.",
        node=lose_confirm_leaf,
        sources=safe_sources(info.role_confirmation_sources),
        additional_instruction="Verify the QB's starting role for the Super Bowl LX losing team."
    )
    add_url_existence_leaf(
        evaluator, lose_confirm_group, "Losing_QB_Team_URL",
        "Reference URL confirming QB role and team",
        urls=info.role_confirmation_sources,
        critical=True
    )

    # 2025 regular season performance (critical parallel)
    perf_group = evaluator.add_parallel(
        id="Losing_QB_Regular_Season_Performance",
        desc="Losing team QB's 2025 regular season statistics",
        parent=lose_qb_node,
        critical=False
    )
    # Passing yards ≥4000
    pass_y_leaf = evaluator.add_leaf(
        id="Passing_Yards",
        desc="2025 regular season passing yards (must be ≥4,000 yards)",
        parent=perf_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2025, {safe_text(info.name, 'the QB')} threw for at least 4,000 passing yards.",
        node=pass_y_leaf,
        sources=safe_sources(info.reg_passing_yards_sources),
        additional_instruction="Confirm the QB's 2025 passing yards are ≥4,000. Allow minor rounding above the threshold."
    )
    add_url_existence_leaf(
        evaluator, perf_group, "Passing_Yards_URL",
        "Reference URL for passing yards statistic",
        urls=info.reg_passing_yards_sources,
        critical=True
    )

    # Passing TDs ≥30
    pass_td_leaf = evaluator.add_leaf(
        id="Passing_Touchdowns",
        desc="2025 regular season passing touchdowns (must be ≥30)",
        parent=perf_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2025, {safe_text(info.name, 'the QB')} threw at least 30 passing touchdowns.",
        node=pass_td_leaf,
        sources=safe_sources(info.reg_passing_tds_sources),
        additional_instruction="Confirm the QB's 2025 passing TDs are ≥30."
    )
    add_url_existence_leaf(
        evaluator, perf_group, "Passing_Touchdowns_URL",
        "Reference URL for passing touchdowns statistic",
        urls=info.reg_passing_tds_sources,
        critical=True
    )

    # Interceptions (non-critical)
    int_leaf = evaluator.add_leaf(
        id="Interceptions",
        desc="2025 regular season interceptions",
        parent=perf_group,
        critical=False
    )
    int_txt = safe_text(info.reg_interceptions, "")
    claim_int = f"In 2025, {safe_text(info.name, 'the QB')} threw {int_txt} interceptions." if int_txt else f"In 2025, {safe_text(info.name, 'the QB')} recorded interceptions."
    await evaluator.verify(
        claim=claim_int,
        node=int_leaf,
        sources=safe_sources(info.reg_interceptions_sources),
        additional_instruction="Verify the interceptions total for the QB in the 2025 regular season."
    )
    add_url_existence_leaf(
        evaluator, perf_group, "Interceptions_URL",
        "Reference URL for interceptions statistic",
        urls=info.reg_interceptions_sources,
        critical=True
    )

    # College background
    lose_college_group = evaluator.add_parallel(
        id="Losing_QB_College_Background",
        desc="College football information for the losing team QB",
        parent=lose_qb_node,
        critical=False
    )
    # College university (critical)
    lose_col_univ_leaf = evaluator.add_leaf(
        id="Losing_QB_College_University",
        desc="University where the QB played college football",
        parent=lose_college_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{safe_text(info.name, 'The QB')} played college football at {safe_text(info.college_university, 'the stated university')}.",
        node=lose_col_univ_leaf,
        sources=safe_sources(info.college_university_sources),
        additional_instruction="Verify college affiliation using reliable sources."
    )
    add_url_existence_leaf(
        evaluator, lose_college_group, "Losing_QB_College_University_URL",
        "Reference URL confirming college affiliation",
        urls=info.college_university_sources,
        critical=True
    )

    # College season year (critical)
    lose_col_year_leaf = evaluator.add_leaf(
        id="Losing_QB_College_Season_Year",
        desc="Specific college season year for statistical reference",
        parent=lose_college_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The relevant college season year for {safe_text(info.name, 'the QB')} is {safe_text(info.college_season_year, 'the stated year')}.",
        node=lose_col_year_leaf,
        sources=safe_sources(info.college_season_year_sources),
        additional_instruction="Verify the season year referenced for the player's college statistics."
    )
    add_url_existence_leaf(
        evaluator, lose_college_group, "Losing_QB_College_Season_Year_URL",
        "Reference URL confirming the season year",
        urls=info.college_season_year_sources,
        critical=True
    )

    # College passing yards ≥3500 (critical)
    lose_col_pyards_leaf = evaluator.add_leaf(
        id="Losing_QB_College_Passing_Yards",
        desc="Passing yards in the specified college season (must be ≥3,500 yards)",
        parent=lose_college_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {safe_text(info.college_season_year, 'the stated year')}, {safe_text(info.name, 'the QB')} threw for at least 3,500 passing yards in college.",
        node=lose_col_pyards_leaf,
        sources=safe_sources(info.college_passing_yards_sources),
        additional_instruction="Confirm the college passing yards are ≥3,500 for that season."
    )
    add_url_existence_leaf(
        evaluator, lose_college_group, "Losing_QB_College_Passing_Yards_URL",
        "Reference URL for college passing yards",
        urls=info.college_passing_yards_sources,
        critical=True
    )

    # College passing touchdowns (non-critical)
    lose_col_ptds_leaf = evaluator.add_leaf(
        id="Losing_QB_College_Passing_Touchdowns",
        desc="Passing touchdowns in the specified college season",
        parent=lose_college_group,
        critical=False
    )
    cptd_txt = safe_text(info.college_passing_tds, "")
    claim_cptd = f"In {safe_text(info.college_season_year, 'the stated year')}, {safe_text(info.name, 'the QB')} threw {cptd_txt} passing touchdowns in college." if cptd_txt else f"In {safe_text(info.college_season_year, 'the stated year')}, {safe_text(info.name, 'the QB')} threw passing touchdowns in college."
    await evaluator.verify(
        claim=claim_cptd,
        node=lose_col_ptds_leaf,
        sources=safe_sources(info.college_passing_tds_sources),
        additional_instruction="Verify the player's college passing touchdowns for that season."
    )
    add_url_existence_leaf(
        evaluator, lose_college_group, "Losing_QB_College_Passing_Touchdowns_URL",
        "Reference URL for college passing touchdowns",
        urls=info.college_passing_tds_sources,
        critical=True
    )

    # Playoff path (sequential)
    lose_path = evaluator.add_sequential(
        id="Losing_QB_Playoff_Path",
        desc="Losing team's playoff path to Super Bowl LX",
        parent=lose_qb_node,
        critical=False
    )
    # Regular season record
    add_url_existence_leaf(
        evaluator, lose_path, "Patriots_Regular_Season_URL",
        "Reference URL for losing team's regular season record",
        urls=info.losing_team_regular_season_sources,
        critical=True
    )
    lose_reg_leaf = evaluator.add_leaf(
        id="Patriots_Regular_Season_Record",
        desc="Losing team's 2025 regular season record",
        parent=lose_path,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2025, the {safe_text(info.team, 'losing team')} had {safe_text(info.losing_team_regular_season_record, 'the stated record')}.",
        node=lose_reg_leaf,
        sources=safe_sources(info.losing_team_regular_season_sources),
        additional_instruction="Verify the regular season record for the losing team."
    )

    # Conference championship
    add_url_existence_leaf(
        evaluator, lose_path, "AFC_Championship_URL",
        "Reference URL for conference championship game",
        urls=info.conf_champ_sources,
        critical=True
    )
    lose_cc_leaf = evaluator.add_leaf(
        id="AFC_Championship_Result",
        desc="Conference championship opponent, score, and date",
        parent=lose_path,
        critical=True
    )
    await evaluator.verify(
        claim=f"In the Conference Championship, the {safe_text(info.team, 'losing team')} played {safe_text(info.conf_champ_opponent_score_date, 'the stated opponent/score/date')}.",
        node=lose_cc_leaf,
        sources=safe_sources(info.conf_champ_sources),
        additional_instruction="Verify the opponent, score, and date for the losing team's conference championship game."
    )

async def verify_game_details(evaluator: Evaluator, parent_node, info: Optional[GameDetails]) -> None:
    game_node = evaluator.add_parallel(
        id="Super_Bowl_Game_Details",
        desc="Basic information about Super Bowl LX game and venue",
        parent=parent_node,
        critical=False
    )
    if not info:
        add_url_existence_leaf(
            evaluator, game_node, "Game_Date_URL",
            "Reference URL confirming game date", urls=[], critical=True
        )
        add_url_existence_leaf(
            evaluator, game_node, "Game_Venue_URL",
            "Reference URL confirming game venue", urls=[], critical=True
        )
        add_url_existence_leaf(
            evaluator, game_node, "Final_Score_URL",
            "Reference URL confirming final score", urls=[], critical=True
        )
        return

    # Game date (critical)
    date_leaf = evaluator.add_leaf(
        id="Game_Date",
        desc="Date of Super Bowl LX",
        parent=game_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Super Bowl LX took place on {safe_text(info.date, 'February 8, 2026')}.",
        node=date_leaf,
        sources=safe_sources(info.date_sources),
        additional_instruction="Verify the exact date of Super Bowl LX."
    )
    add_url_existence_leaf(
        evaluator, game_node, "Game_Date_URL",
        "Reference URL confirming game date",
        urls=info.date_sources,
        critical=True
    )

    # Venue (critical)
    venue_leaf = evaluator.add_leaf(
        id="Game_Venue",
        desc="Venue where Super Bowl LX was played including location",
        parent=game_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Super Bowl LX was played at {safe_text(info.venue_name_and_location, 'the stated venue and location')}.",
        node=venue_leaf,
        sources=safe_sources(info.venue_sources),
        additional_instruction="Verify the venue name and location for Super Bowl LX."
    )
    add_url_existence_leaf(
        evaluator, game_node, "Game_Venue_URL",
        "Reference URL confirming game venue",
        urls=info.venue_sources,
        critical=True
    )

    # Final score (critical)
    score_leaf = evaluator.add_leaf(
        id="Final_Score",
        desc="Final score of Super Bowl LX",
        parent=game_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The final score of Super Bowl LX was {safe_text(info.final_score, 'the stated final score')}.",
        node=score_leaf,
        sources=safe_sources(info.final_score_sources),
        additional_instruction="Verify the final score of Super Bowl LX, including which team won."
    )
    add_url_existence_leaf(
        evaluator, game_node, "Final_Score_URL",
        "Reference URL confirming final score",
        urls=info.final_score_sources,
        critical=True
    )

    # Stadium capacity (non-critical)
    capacity_leaf = evaluator.add_leaf(
        id="Stadium_Capacity",
        desc="Stadium seating capacity information",
        parent=game_node,
        critical=False
    )
    cap_txt = safe_text(info.stadium_capacity, "")
    claim_cap = f"The stadium seating capacity is {cap_txt}." if cap_txt else "The stadium seating capacity is reported."
    await evaluator.verify(
        claim=claim_cap,
        node=capacity_leaf,
        sources=safe_sources(info.stadium_capacity_sources),
        additional_instruction="Verify the stated stadium seating capacity; allow minor differences if ranges or multiple configurations are noted."
    )
    add_url_existence_leaf(
        evaluator, game_node, "Stadium_Capacity_URL",
        "Reference URL for stadium capacity information",
        urls=info.stadium_capacity_sources,
        critical=True
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
    Evaluate an answer for the Super Bowl LX key players analysis task.
    """
    # Initialize evaluator (root as parallel, non-critical to allow mixed criticality children)
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

    # Extract all structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_sb_key_players(),
        template_class=SBKeyPlayersExtraction,
        extraction_name="sb_key_players_extraction"
    )

    # Build verification tree according to rubric
    await verify_mvp_rb(evaluator, root, extracted.mvp_rb)
    await verify_defensive_cb(evaluator, root, extracted.defensive_cb)
    await verify_winning_qb(evaluator, root, extracted.winning_qb)
    await verify_losing_qb(evaluator, root, extracted.losing_qb)
    await verify_game_details(evaluator, root, extracted.game)

    # Return structured result
    return evaluator.get_summary()