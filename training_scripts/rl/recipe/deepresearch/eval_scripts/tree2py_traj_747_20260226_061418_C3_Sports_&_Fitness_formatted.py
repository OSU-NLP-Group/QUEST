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
TASK_ID = "acc_freshman_record_2024_25"
TASK_DESCRIPTION = (
    "During the 2024-25 NCAA Division I men's college basketball season, a freshman player set a new Atlantic Coast "
    "Conference (ACC) single-game scoring record for freshmen. Your task is to identify this player and provide "
    "comprehensive information about their record-breaking performance and subsequent achievements.\n\n"
    "Please provide the following information:\n\n"
    "1. Player Identification: The player's full name and the university they represented.\n\n"
    "2. Scoring Record Details:\n"
    "- The exact point total scored in the record-setting game\n"
    "- The opposing team in that game\n"
    "- The date the record was set\n\n"
    "3. National Awards: Identify at least two major national player of the year awards the player won during the "
    "2024-25 season. Specifically include:\n"
    "- The Oscar Robertson Trophy (USBWA National Player of the Year)\n"
    "- The Wooden Award\n\n"
    "4. NBA Draft Information: Provide the player's 2025 NBA Draft information:\n"
    "- Draft position (overall pick number)\n"
    "- The NBA team that selected them\n\n"
    "For each piece of information provided, include at least one reference URL from official sources (such as "
    "university athletics pages, conference websites, sports news organizations, or official NBA sources) that confirms the information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PlayerSection(BaseModel):
    full_name: Optional[str] = None
    university: Optional[str] = None
    name_urls: List[str] = Field(default_factory=list)
    university_urls: List[str] = Field(default_factory=list)


class RecordSection(BaseModel):
    claims_acc_freshman_record: Optional[bool] = None
    record_urls: List[str] = Field(default_factory=list)
    point_total: Optional[str] = None
    point_urls: List[str] = Field(default_factory=list)
    opponent: Optional[str] = None
    opponent_urls: List[str] = Field(default_factory=list)
    game_date: Optional[str] = None
    date_urls: List[str] = Field(default_factory=list)


class AwardsSection(BaseModel):
    oscar_won: Optional[bool] = None
    oscar_urls: List[str] = Field(default_factory=list)
    wooden_won: Optional[bool] = None
    wooden_urls: List[str] = Field(default_factory=list)


class DraftSection(BaseModel):
    draft_pick_overall: Optional[str] = None
    draft_pick_urls: List[str] = Field(default_factory=list)
    draft_team: Optional[str] = None
    draft_team_urls: List[str] = Field(default_factory=list)


class ExtractionAll(BaseModel):
    player: Optional[PlayerSection] = None
    record: Optional[RecordSection] = None
    awards: Optional[AwardsSection] = None
    draft: Optional[DraftSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    You must extract structured information from the answer related to the ACC freshman single-game scoring record
    set during the 2024–25 season, the player's identity, the required national awards, and the 2025 NBA Draft info.
    Extract ONLY what is explicitly present in the answer. Do not invent anything.

    Return a JSON object with the following structure:

    {
      "player": {
        "full_name": string or null,
        "university": string or null,
        "name_urls": [array of URLs explicitly present in the answer that support the player's identity/name],
        "university_urls": [array of URLs explicitly present in the answer that support the player's university affiliation]
      },
      "record": {
        "claims_acc_freshman_record": boolean or null,  // whether the answer explicitly states it set the ACC freshman single-game scoring record
        "record_urls": [array of URLs that support the 'ACC freshman single-game scoring record' claim],
        "point_total": string or null,  // exact points (keep as text if formatted like '35' or '35 points')
        "point_urls": [array of URLs that support the stated point total],
        "opponent": string or null,
        "opponent_urls": [array of URLs that support the stated opponent],
        "game_date": string or null,  // date text as provided in the answer (e.g., 'Jan. 12, 2025' or 'January 12, 2025')
        "date_urls": [array of URLs that support the stated date]
      },
      "awards": {
        "oscar_won": boolean or null,  // whether the answer explicitly states the player won the Oscar Robertson Trophy in 2024–25
        "oscar_urls": [array of URLs supporting the Oscar Robertson Trophy win],
        "wooden_won": boolean or null, // whether the answer explicitly states the player won the Wooden Award in 2024–25
        "wooden_urls": [array of URLs supporting the Wooden Award win]
      },
      "draft": {
        "draft_pick_overall": string or null, // the overall pick number as stated (e.g., '1st overall', 'No. 3', '3')
        "draft_pick_urls": [array of URLs supporting the overall pick number],
        "draft_team": string or null, // the NBA team that selected the player
        "draft_team_urls": [array of URLs supporting the drafting team]
      }
    }

    Rules:
    - Extract only URLs that are explicitly present in the answer (including markdown links); do not infer URLs.
    - If any field is not explicitly present, return null or an empty array accordingly.
    - Do not transform or normalize dates/numbers; keep the exact text from the answer.
    - Deduplicate URLs and ensure they are valid URLs with a protocol (http:// or https://).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in lists:
        for u in _nonempty_urls(lst):
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _text_or_blank(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_player_identification(
    evaluator: Evaluator,
    parent_node,
    data: ExtractionAll,
) -> None:
    player_node = evaluator.add_parallel(
        id="Player_Identification",
        desc="Provide the player's full name and university, with supporting URLs.",
        parent=parent_node,
        critical=True,
    )

    full_name = _text_or_blank(getattr(data.player, "full_name", None) if data.player else None)
    university = _text_or_blank(getattr(data.player, "university", None) if data.player else None)
    name_urls = _nonempty_urls(getattr(data.player, "name_urls", []) if data.player else [])
    univ_urls = _nonempty_urls(getattr(data.player, "university_urls", []) if data.player else [])

    # Existence checks
    evaluator.add_custom_node(
        result=bool(full_name),
        id="Player_Full_Name_Provided",
        desc="Player's full name is provided.",
        parent=player_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(university),
        id="University_Provided",
        desc="University represented is provided.",
        parent=player_node,
        critical=True,
    )

    # URL support: name
    name_support_node = evaluator.add_leaf(
        id="Player_Name_Has_Supporting_URL",
        desc="At least one official/reputable reference URL supports the player's identity (name).",
        parent=player_node,
        critical=True,
    )
    name_claim = f"The page confirms the player's full name is '{full_name}'." if full_name else "The page confirms the player's full name."
    await evaluator.verify(
        claim=name_claim,
        node=name_support_node,
        sources=name_urls,
        additional_instruction="Prefer official sources (university, conference, reputable sports orgs). Minor variations in punctuation/casing are acceptable.",
    )

    # URL support: university affiliation
    univ_support_node = evaluator.add_leaf(
        id="Player_University_Has_Supporting_URL",
        desc="At least one official/reputable reference URL supports the player's university affiliation.",
        parent=player_node,
        critical=True,
    )
    univ_claim = (
        f"The page confirms that {full_name} represented {university} in men's college basketball."
        if full_name and university
        else "The page confirms the player's university affiliation."
    )
    await evaluator.verify(
        claim=univ_claim,
        node=univ_support_node,
        sources=univ_urls,
        additional_instruction="Prefer official university athletics, conference sites, or major reputable outlets.",
    )


async def build_record_details(
    evaluator: Evaluator,
    parent_node,
    data: ExtractionAll,
) -> None:
    rec_node = evaluator.add_parallel(
        id="Record_Setting_Game_Details",
        desc="Provide the record-setting game details (points, opponent, date) and that it set the ACC freshman single-game scoring record, with supporting URLs.",
        parent=parent_node,
        critical=True,
    )

    full_name = _text_or_blank(getattr(data.player, "full_name", None) if data.player else None)

    # Extracted fields
    rec = data.record or RecordSection()
    point_total = _text_or_blank(rec.point_total)
    opponent = _text_or_blank(rec.opponent)
    game_date = _text_or_blank(rec.game_date)

    record_urls = _nonempty_urls(rec.record_urls)
    point_urls = _merge_sources(rec.point_urls, rec.record_urls)
    opponent_urls = _merge_sources(rec.opponent_urls, rec.record_urls)
    date_urls = _merge_sources(rec.date_urls, rec.record_urls)

    # States it is ACC freshman single-game scoring record (from the answer content)
    evaluator.add_custom_node(
        result=bool(rec.claims_acc_freshman_record is True),
        id="States_It_Is_ACC_Freshman_Single_Game_Scoring_Record",
        desc="Answer explicitly indicates the performance set a new ACC single-game scoring record for freshmen.",
        parent=rec_node,
        critical=True,
    )

    # Record claim supported by URL(s)
    rec_support_node = evaluator.add_leaf(
        id="Record_Claim_Has_Supporting_URL",
        desc="At least one official/reputable reference URL supports the claim that the performance set the ACC freshman single-game scoring record.",
        parent=rec_node,
        critical=True,
    )
    rec_claim = (
        f"{full_name} set the ACC single-game scoring record for freshmen."
        if full_name else
        "This performance set the ACC single-game scoring record for freshmen."
    )
    await evaluator.verify(
        claim=rec_claim,
        node=rec_support_node,
        sources=record_urls,
        additional_instruction="Check the page text to confirm the phrase 'ACC freshman single-game scoring record' or an equivalent explicit statement.",
    )

    # Point total provided
    evaluator.add_custom_node(
        result=bool(point_total),
        id="Point_Total_Provided",
        desc="Exact point total scored in the record-setting game is provided.",
        parent=rec_node,
        critical=True,
    )
    # Point total supported
    points_support_node = evaluator.add_leaf(
        id="Point_Total_Has_Supporting_URL",
        desc="At least one official/reputable reference URL supports the stated point total for the record-setting game.",
        parent=rec_node,
        critical=True,
    )
    pt_claim = (
        f"In the record-setting game, {full_name} scored {point_total} points."
        if full_name and point_total else
        f"The record-setting game featured a point total of {point_total} points." if point_total else
        "The record-setting game's point total is supported."
    )
    await evaluator.verify(
        claim=pt_claim,
        node=points_support_node,
        sources=point_urls,
        additional_instruction="Verify the exact points scored in that specific record-setting game; allow minor formatting differences (e.g., 'points' label).",
    )

    # Opponent provided
    evaluator.add_custom_node(
        result=bool(opponent),
        id="Opponent_Provided",
        desc="Opposing team in the record-setting game is provided.",
        parent=rec_node,
        critical=True,
    )
    # Opponent supported
    opp_support_node = evaluator.add_leaf(
        id="Opponent_Has_Supporting_URL",
        desc="At least one official/reputable reference URL supports the stated opponent for the record-setting game.",
        parent=rec_node,
        critical=True,
    )
    opp_claim = f"The record-setting game was against {opponent}." if opponent else "The record-setting game's opponent is supported."
    await evaluator.verify(
        claim=opp_claim,
        node=opp_support_node,
        sources=opponent_urls,
        additional_instruction="Check the game recap, box score, or official source confirming the opponent.",
    )

    # Game date provided
    evaluator.add_custom_node(
        result=bool(game_date),
        id="Game_Date_Provided",
        desc="Date the record was set is provided.",
        parent=rec_node,
        critical=True,
    )
    # Game date supported
    date_support_node = evaluator.add_leaf(
        id="Game_Date_Has_Supporting_URL",
        desc="At least one official/reputable reference URL supports the stated date for the record-setting game.",
        parent=rec_node,
        critical=True,
    )
    date_claim = f"The record-setting game occurred on {game_date}." if game_date else "The record-setting game's date is supported."
    await evaluator.verify(
        claim=date_claim,
        node=date_support_node,
        sources=date_urls,
        additional_instruction="Allow reasonable date formatting variations; confirm the same calendar date.",
    )


async def build_awards(
    evaluator: Evaluator,
    parent_node,
    data: ExtractionAll,
) -> None:
    awards_node = evaluator.add_parallel(
        id="National_Awards_2024_25",
        desc="Provide at least two major national player-of-the-year awards the player won during 2024–25, specifically including the Oscar Robertson Trophy and the Wooden Award, each with supporting URLs.",
        parent=parent_node,
        critical=True,
    )

    full_name = _text_or_blank(getattr(data.player, "full_name", None) if data.player else None)
    aw = data.awards or AwardsSection()

    # Oscar stated in answer
    evaluator.add_custom_node(
        result=bool(aw.oscar_won is True),
        id="Oscar_Robertson_Trophy_Included_As_Win",
        desc="States the player won the Oscar Robertson Trophy (USBWA National Player of the Year) in the 2024–25 season.",
        parent=awards_node,
        critical=True,
    )
    # Oscar supported by URL
    oscar_support_node = evaluator.add_leaf(
        id="Oscar_Robertson_Trophy_Has_Supporting_URL",
        desc="At least one official/reputable reference URL supports the Oscar Robertson Trophy win.",
        parent=awards_node,
        critical=True,
    )
    oscar_claim = (
        f"{full_name} won the Oscar Robertson Trophy (USBWA National Player of the Year) for the 2024–25 season."
        if full_name else
        "The player won the Oscar Robertson Trophy (USBWA National Player of the Year) for the 2024–25 season."
    )
    await evaluator.verify(
        claim=oscar_claim,
        node=oscar_support_node,
        sources=_nonempty_urls(aw.oscar_urls),
        additional_instruction="Prefer official USBWA announcement, university releases, or major reputable outlets.",
    )

    # Wooden stated in answer
    evaluator.add_custom_node(
        result=bool(aw.wooden_won is True),
        id="Wooden_Award_Included_As_Win",
        desc="States the player won the Wooden Award in the 2024–25 season timeframe (as asked).",
        parent=awards_node,
        critical=True,
    )
    # Wooden supported by URL
    wooden_support_node = evaluator.add_leaf(
        id="Wooden_Award_Has_Supporting_URL",
        desc="At least one official/reputable reference URL supports the Wooden Award win.",
        parent=awards_node,
        critical=True,
    )
    wooden_claim = (
        f"{full_name} won the Wooden Award for the 2024–25 season."
        if full_name else
        "The player won the Wooden Award for the 2024–25 season."
    )
    await evaluator.verify(
        claim=wooden_claim,
        node=wooden_support_node,
        sources=_nonempty_urls(aw.wooden_urls),
        additional_instruction="Prefer official Wooden Award site announcements, university releases, or major reputable outlets.",
    )


async def build_draft(
    evaluator: Evaluator,
    parent_node,
    data: ExtractionAll,
) -> None:
    draft_node = evaluator.add_parallel(
        id="NBA_Draft_2025",
        desc="Provide the player's 2025 NBA Draft position and drafting team, each with supporting URLs.",
        parent=parent_node,
        critical=True,
    )

    full_name = _text_or_blank(getattr(data.player, "full_name", None) if data.player else None)
    dr = data.draft or DraftSection()

    pick_text = _text_or_blank(dr.draft_pick_overall)
    team_text = _text_or_blank(dr.draft_team)
    pick_urls = _nonempty_urls(dr.draft_pick_urls)
    team_urls = _nonempty_urls(dr.draft_team_urls)

    # Draft position provided
    evaluator.add_custom_node(
        result=bool(pick_text),
        id="Draft_Position_Provided",
        desc="Overall pick number (draft position) is provided.",
        parent=draft_node,
        critical=True,
    )
    # Draft position supported
    draft_pos_support = evaluator.add_leaf(
        id="Draft_Position_Has_Supporting_URL",
        desc="At least one official NBA (or equivalently authoritative) source URL supports the stated draft position.",
        parent=draft_node,
        critical=True,
    )
    pick_claim = (
        f"In the 2025 NBA Draft, {full_name} was selected {pick_text} overall."
        if full_name and pick_text else
        f"In the 2025 NBA Draft, the player was selected {pick_text} overall." if pick_text else
        "The player's 2025 NBA Draft overall pick is supported."
    )
    await evaluator.verify(
        claim=pick_claim,
        node=draft_pos_support,
        sources=_merge_sources(pick_urls, team_urls),
        additional_instruction="Prefer official NBA.com draft tracker, team press releases, or other highly authoritative sources.",
    )

    # Drafting team provided
    evaluator.add_custom_node(
        result=bool(team_text),
        id="Drafting_Team_Provided",
        desc="NBA team that selected the player is provided.",
        parent=draft_node,
        critical=True,
    )
    # Drafting team supported
    draft_team_support = evaluator.add_leaf(
        id="Drafting_Team_Has_Supporting_URL",
        desc="At least one official NBA (or equivalently authoritative) source URL supports the stated drafting team.",
        parent=draft_node,
        critical=True,
    )
    team_claim = (
        f"In the 2025 NBA Draft, {full_name} was selected by the {team_text}."
        if full_name and team_text else
        f"In the 2025 NBA Draft, the player was selected by the {team_text}." if team_text else
        "The player's 2025 NBA Draft team is supported."
    )
    await evaluator.verify(
        claim=team_claim,
        node=draft_team_support,
        sources=_merge_sources(team_urls, pick_urls),
        additional_instruction="Prefer official NBA.com, team sites, or draft trackers from authoritative outlets.",
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
    # Initialize evaluator with a root; we add a critical top-level task node under it.
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=ExtractionAll,
        extraction_name="extracted_acc_freshman_record_package",
    )

    # Build critical investigation node (reflecting rubric root)
    top = evaluator.add_parallel(
        id="ACC_Freshman_Record_Investigation",
        desc="Identify the ACC freshman single-game scoring record setter in the 2024–25 season and provide record details, required awards, NBA draft info, and per-field supporting URLs as requested.",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_player_identification(evaluator, top, extracted)
    await build_record_details(evaluator, top, extracted)
    await build_awards(evaluator, top, extracted)
    await build_draft(evaluator, top, extracted)

    # Return summary
    return evaluator.get_summary()