import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "coach_pac12_2026_hire"
TASK_DESCRIPTION = (
    "Identify the college football head coach who was hired between September 2024 and January 2026 to lead a program "
    "joining the Pac-12 Conference in 2026, and who meets all of the following career requirements: has served as an "
    "FBS head coach for at least 10 seasons total, achieved back-to-back nine-win seasons during their career, led a "
    "team to at least three bowl game appearances at a single institution, maintains a career winning percentage above "
    ".500, and previously served as a head coach in a major conference (Pac-12, Big Ten, SEC, ACC, or Big 12)."
)

PAC12_2026_MEMBERS = [
    "Boise State",
    "Colorado State",
    "Fresno State",
    "Oregon State",
    "San Diego State",
    "Texas State",
    "Utah State",
    "Washington State",
]

MAJOR_CONFERENCES = ["Pac-12", "Pac 12", "Big Ten", "SEC", "ACC", "Big 12", "Big Twelve"]


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CoachBasic(BaseModel):
    coach_name: Optional[str] = None
    hiring_institution: Optional[str] = None
    hiring_date: Optional[str] = None
    hiring_urls: List[str] = Field(default_factory=list)
    pac12_membership_urls: List[str] = Field(default_factory=list)


class FBSTenure(BaseModel):
    seasons_total: Optional[str] = None
    tenure_summary: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MajorConferenceExp(BaseModel):
    institution: Optional[str] = None
    conference: Optional[str] = None
    years: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SeasonInfo(BaseModel):
    year: Optional[str] = None
    wins: Optional[str] = None
    team: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class NineWinSeasons(BaseModel):
    first: Optional[SeasonInfo] = None
    second: Optional[SeasonInfo] = None


class BowlAppearances(BaseModel):
    institution: Optional[str] = None
    bowl_count: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CareerRecord(BaseModel):
    wins: Optional[str] = None
    losses: Optional[str] = None
    win_pct: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CoachExtraction(BaseModel):
    basic: Optional[CoachBasic] = None
    fbs_tenure: Optional[FBSTenure] = None
    major_conf: Optional[MajorConferenceExp] = None
    nine_win: Optional[NineWinSeasons] = None
    bowls: Optional[BowlAppearances] = None
    career: Optional[CareerRecord] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_info() -> str:
    return """
Extract the key information about the identified college football head coach from the answer.

You must extract exactly and only what is present in the answer. Do not infer or invent.

Return a JSON with the following structure:

{
  "basic": {
    "coach_name": string or null,
    "hiring_institution": string or null,
    "hiring_date": string or null,  // exact date or month-year as written
    "hiring_urls": [list of URLs explicitly mentioned for the hiring announcement/date],
    "pac12_membership_urls": [list of URLs explicitly mentioned that document the institution's Pac-12 2026 membership]
  },
  "fbs_tenure": {
    "seasons_total": string or null,         // total FBS head coaching seasons as written (e.g., "12", "12 seasons")
    "tenure_summary": string or null,        // summary or list of institutions/years if provided
    "urls": [list of URLs cited to document career experience/tenure]
  },
  "major_conf": {
    "institution": string or null,           // institution where the coach previously served as head coach in a major conference
    "conference": string or null,            // name of the conference (e.g., "SEC", "Big Ten", etc.)
    "years": string or null,                 // years of that head coaching stint, if mentioned
    "urls": [list of URLs documenting this prior major-conference head coaching role]
  },
  "nine_win": {
    "first": {
      "year": string or null,                // year of the first nine-win season
      "wins": string or null,                // wins in that year (e.g., "9", "10")
      "team": string or null,                // team/school for that season
      "urls": [list of URLs documenting this season]
    },
    "second": {
      "year": string or null,                // year of the second consecutive nine-win season
      "wins": string or null,                // wins in that year
      "team": string or null,                // team/school for that season
      "urls": [list of URLs documenting this season]
    }
  },
  "bowls": {
    "institution": string or null,           // the single institution where at least three bowl appearances were achieved
    "bowl_count": string or null,            // number of bowl appearances at that institution as written
    "urls": [list of URLs documenting these bowl appearances]
  },
  "career": {
    "wins": string or null,                  // total career head coaching wins as written
    "losses": string or null,                // total career head coaching losses as written
    "win_pct": string or null,               // career winning percentage as written (e.g., ".612", "61.2%")
    "urls": [list of URLs documenting career record and percentage]
  }
}

Rules for URLs:
- Include only actual URLs explicitly present in the answer (plain or markdown links).
- Do not fabricate or infer URLs.
- If no URLs are provided for a field, return an empty list for that URLs field.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if u and isinstance(u, str):
                if u not in seen:
                    seen.add(u)
                    out.append(u)
    return out


def _safe_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    try:
        # extract first integer substring
        import re
        m = re.search(r"-?\d+", text.replace(",", ""))
        if not m:
            return None
        return int(m.group(0))
    except Exception:
        return None


def _coach_or_generic(basic: Optional[CoachBasic]) -> str:
    return basic.coach_name if (basic and basic.coach_name) else "the coach"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_career_experience_verification(evaluator: Evaluator, parent, data: CoachExtraction):
    node = evaluator.add_parallel(
        id="Career_Experience_Verification",
        desc="Verifies the coach's total career experience meets the minimum requirements",
        parent=parent,
        critical=False
    )

    # FBS Head Coaching Tenure (Sequential, Critical)
    fbs_node = evaluator.add_sequential(
        id="FBS_Head_Coaching_Tenure",
        desc="Confirms the coach has served as an FBS head coach for at least 10 seasons total across all positions",
        parent=node,
        critical=True
    )

    fbs = data.fbs_tenure or FBSTenure()
    career = data.career or CareerRecord()
    urls_fbs = _combine_urls(fbs.urls, career.urls)
    coach_name = _coach_or_generic(data.basic)

    # Total_Seasons_Count (leaf)
    total_seasons_leaf = evaluator.add_leaf(
        id="Total_Seasons_Count",
        desc="Verifies the total number of seasons as FBS head coach equals or exceeds 10",
        parent=fbs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} has served at least 10 seasons as an FBS head coach in total across all head coaching stints.",
        node=total_seasons_leaf,
        sources=urls_fbs,
        additional_instruction="Use the provided sources (bios, career summaries, sports-reference pages) to tally FBS head coaching seasons across all institutions and verify the total is >= 10. Count only FBS (Division I-A/FBS) seasons."
    )

    # FBS_Level_Confirmation (leaf)
    fbs_level_leaf = evaluator.add_leaf(
        id="FBS_Level_Confirmation",
        desc="Confirms all counted seasons were at FBS-level institutions",
        parent=fbs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"All seasons counted toward {coach_name}'s head coaching tenure used for the total were at FBS (Division I-A) institutions.",
        node=fbs_level_leaf,
        sources=urls_fbs,
        additional_instruction="Confirm that each head coaching season counted occurred at an FBS (formerly Division I-A) program during those years."
    )

    # Career_Experience_URL (existence check)
    evaluator.add_custom_node(
        result=len(urls_fbs) > 0,
        id="Career_Experience_URL",
        desc="Provides URL documentation for the coach's career experience history",
        parent=fbs_node,
        critical=True
    )

    # Major Conference Background (Sequential, Critical)
    mc_node = evaluator.add_sequential(
        id="Major_Conference_Background",
        desc="Confirms the coach previously served as a head coach in a major conference (Pac-12, Big Ten, SEC, ACC, or Big 12)",
        parent=node,
        critical=True
    )

    mc = data.major_conf or MajorConferenceExp()
    mc_urls = _combine_urls(mc.urls)

    # Previous_Major_Conference_Position
    prev_mc_leaf = evaluator.add_leaf(
        id="Previous_Major_Conference_Position",
        desc="Identifies at least one previous head coaching position in a major conference",
        parent=mc_node,
        critical=True
    )
    inst_txt = mc.institution or "an institution"
    conf_txt = mc.conference or "a major conference"
    await evaluator.verify(
        claim=f"{coach_name} previously served as a head coach at {inst_txt} in the {conf_txt}.",
        node=prev_mc_leaf,
        sources=mc_urls,
        additional_instruction="Verify that the role was HEAD COACH (not assistant) and that the conference listed corresponds to that institution during the tenure."
    )

    # Conference_Classification_Verification (simple check)
    conf_class_leaf = evaluator.add_leaf(
        id="Conference_Classification_Verification",
        desc="Confirms the identified conference is classified as a major/power conference",
        parent=mc_node,
        critical=True
    )
    is_major_list = ", ".join(MAJOR_CONFERENCES)
    await evaluator.verify(
        claim=f"The conference '{mc.conference}' is one of the major conferences: {is_major_list}.",
        node=conf_class_leaf,
        additional_instruction="Accept minor naming variations (e.g., 'Pac-12' vs 'Pac 12'). Focus on membership of Pac-12, Big Ten, SEC, ACC, or Big 12."
    )

    # Major_Conference_URL (existence)
    evaluator.add_custom_node(
        result=len(mc_urls) > 0,
        id="Major_Conference_URL",
        desc="Provides URL documentation for the coach's major conference experience",
        parent=mc_node,
        critical=True
    )


async def build_performance_record_verification(evaluator: Evaluator, parent, data: CoachExtraction):
    node = evaluator.add_parallel(
        id="Performance_Record_Verification",
        desc="Verifies the coach's performance achievements meet all specified criteria",
        parent=parent,
        critical=False
    )

    coach_name = _coach_or_generic(data.basic)

    # Nine_Win_Seasons_Achievement (Sequential, Critical)
    nine_node = evaluator.add_sequential(
        id="Nine_Win_Seasons_Achievement",
        desc="Confirms the coach achieved back-to-back nine-win seasons at some point in their career",
        parent=node,
        critical=True
    )
    first = (data.nine_win.first if data.nine_win else None) or SeasonInfo()
    second = (data.nine_win.second if data.nine_win else None) or SeasonInfo()
    first_urls = _combine_urls(first.urls)
    second_urls = _combine_urls(second.urls)
    both_urls = _combine_urls(first_urls, second_urls)

    # First_Nine_Win_Season (Parallel, Critical)
    first_node = evaluator.add_parallel(
        id="First_Nine_Win_Season",
        desc="Identifies the first season in the consecutive pair with 9 or more wins",
        parent=nine_node,
        critical=True
    )
    # First_Season_Win_Count
    first_win_leaf = evaluator.add_leaf(
        id="First_Season_Win_Count",
        desc="Verifies the win total equals or exceeds 9 for the first season",
        parent=first_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {first.year}, {coach_name}'s {first.team} won at least 9 games.",
        node=first_win_leaf,
        sources=first_urls,
        additional_instruction="Check official season summaries or records on the provided pages to confirm the win total >= 9."
    )
    # First_Season_Year_Identification
    first_year_leaf = evaluator.add_leaf(
        id="First_Season_Year_Identification",
        desc="Identifies the specific year of the first nine-win season",
        parent=first_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first nine-win season occurred in {first.year} for {coach_name} at {first.team}.",
        node=first_year_leaf,
        sources=first_urls,
        additional_instruction="The pages should explicitly show the season year associated with the nine or more wins."
    )

    # Second_Nine_Win_Season (Parallel, Critical)
    second_node = evaluator.add_parallel(
        id="Second_Nine_Win_Season",
        desc="Identifies the immediately following season with 9 or more wins",
        parent=nine_node,
        critical=True
    )
    # Second_Season_Win_Count
    second_win_leaf = evaluator.add_leaf(
        id="Second_Season_Win_Count",
        desc="Verifies the win total equals or exceeds 9 for the second season",
        parent=second_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In {second.year}, {coach_name}'s {second.team} won at least 9 games.",
        node=second_win_leaf,
        sources=second_urls,
        additional_instruction="Check official season summaries or records on the provided pages to confirm the win total >= 9."
    )
    # Consecutive_Year_Verification
    consecutive_leaf = evaluator.add_leaf(
        id="Consecutive_Year_Verification",
        desc="Confirms the second season immediately follows the first season chronologically",
        parent=second_node,
        critical=True
    )
    first_year_int = _safe_int(first.year)
    second_year_int = _safe_int(second.year)
    if first_year_int is not None and second_year_int is not None:
        await evaluator.verify(
            claim=f"The seasons {first_year_int} and {second_year_int} are consecutive (difference of 1).",
            node=consecutive_leaf,
            additional_instruction="This is a simple logical check on the years provided."
        )
    else:
        await evaluator.verify(
            claim=f"The second nine-win season year ({second.year}) immediately follows the first ({first.year}).",
            node=consecutive_leaf,
            sources=both_urls,
            additional_instruction="If exact years are present on the pages, ensure the second is the next calendar year after the first."
        )

    # Nine_Win_Seasons_URL (existence)
    evaluator.add_custom_node(
        result=len(both_urls) > 0,
        id="Nine_Win_Seasons_URL",
        desc="Provides URL documentation for the back-to-back nine-win seasons",
        parent=nine_node,
        critical=True
    )

    # Bowl_Game_Appearances (Sequential, Critical)
    bowl_node = evaluator.add_sequential(
        id="Bowl_Game_Appearances",
        desc="Confirms the coach led a team to at least three bowl game appearances during tenure at a single institution",
        parent=node,
        critical=True
    )
    bowls = data.bowls or BowlAppearances()
    bowl_urls = _combine_urls(bowls.urls)
    bowl_inst = bowls.institution or "the institution"

    # Institution_Identification
    bowl_inst_leaf = evaluator.add_leaf(
        id="Institution_Identification_Bowls",
        desc="Identifies the single institution where the coach led the team to at least three bowl appearances",
        parent=bowl_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} achieved at least three bowl appearances at {bowl_inst}.",
        node=bowl_inst_leaf,
        sources=bowl_urls,
        additional_instruction="The pages should indicate that the bowl appearances occurred while the coach was head coach at this single institution."
    )

    # Bowl_Appearance_Count (Parallel, Critical)
    bowl_count_node = evaluator.add_parallel(
        id="Bowl_Appearance_Count",
        desc="Verifies the total number of bowl game appearances at that institution equals or exceeds 3",
        parent=bowl_node,
        critical=True
        )
    # Minimum_Three_Bowls
    min_three_leaf = evaluator.add_leaf(
        id="Minimum_Three_Bowls",
        desc="Confirms at least three distinct bowl games were reached",
        parent=bowl_count_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} led {bowl_inst} to at least three bowl game appearances.",
        node=min_three_leaf,
        sources=bowl_urls,
        additional_instruction="Confirm a count of 3 or more distinct bowl games under this coach at this institution."
    )
    # Same_Institution_Verification
    same_inst_leaf = evaluator.add_leaf(
        id="Same_Institution_Verification",
        desc="Confirms all counted bowl appearances occurred during the coach's tenure at the same institution",
        parent=bowl_count_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"All the bowl appearances counted (at least three) occurred during {coach_name}'s tenure at {bowl_inst}.",
        node=same_inst_leaf,
        sources=bowl_urls,
        additional_instruction="Ensure the bowl appearances listed correspond to seasons when the coach was head coach at this institution."
    )

    # Bowl_Appearances_URL (existence)
    evaluator.add_custom_node(
        result=len(bowl_urls) > 0,
        id="Bowl_Appearances_URL",
        desc="Provides URL documentation for the bowl game appearances",
        parent=bowl_node,
        critical=True
    )

    # Career_Winning_Percentage (Sequential, Critical)
    cwp_node = evaluator.add_sequential(
        id="Career_Winning_Percentage",
        desc="Confirms the coach maintains a career winning percentage above .500 across all head coaching positions",
        parent=node,
        critical=True
    )
    career = data.career or CareerRecord()
    career_urls = _combine_urls(career.urls)

    # Total_Win_Loss_Record (Parallel, Critical)
    wl_node = evaluator.add_parallel(
        id="Total_Win_Loss_Record",
        desc="Identifies the coach's complete career win-loss record across all head coaching positions",
        parent=cwp_node,
        critical=True
    )
    # Total_Career_Wins
    wins_leaf = evaluator.add_leaf(
        id="Total_Career_Wins",
        desc="States the total number of career wins",
        parent=wl_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} has {career.wins} career head coaching wins.",
        node=wins_leaf,
        sources=career_urls,
        additional_instruction="Verify the exact number of career wins from the cited record pages."
    )
    # Total_Career_Losses
    losses_leaf = evaluator.add_leaf(
        id="Total_Career_Losses",
        desc="States the total number of career losses",
        parent=wl_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} has {career.losses} career head coaching losses.",
        node=losses_leaf,
        sources=career_urls,
        additional_instruction="Verify the exact number of career losses from the cited record pages."
    )

    # Winning_Percentage_Calculation
    winpct_leaf = evaluator.add_leaf(
        id="Winning_Percentage_Calculation",
        desc="Verifies the winning percentage (wins divided by total games) exceeds 0.500",
        parent=cwp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name}'s career winning percentage exceeds .500.",
        node=winpct_leaf,
        sources=career_urls,
        additional_instruction="Use the record pages to compute or confirm wins/(wins+losses) > 0.500. Accept minor rounding differences."
    )

    # Winning_Percentage_URL (existence)
    evaluator.add_custom_node(
        result=len(career_urls) > 0,
        id="Winning_Percentage_URL",
        desc="Provides URL documentation for the career record and winning percentage",
        parent=cwp_node,
        critical=True
    )


async def build_institutional_context_verification(evaluator: Evaluator, parent, data: CoachExtraction):
    node = evaluator.add_parallel(
        id="Institutional_Context_Verification",
        desc="Verifies the hiring institution and conference affiliation meet all specified criteria",
        parent=parent,
        critical=False
    )

    coach_name = _coach_or_generic(data.basic)
    basic = data.basic or CoachBasic()

    # Pac12_Membership_2026 (Sequential, Critical)
    pac12_node = evaluator.add_sequential(
        id="Pac12_Membership_2026",
        desc="Confirms the hiring institution is among the football-playing members joining the Pac-12 Conference in 2026",
        parent=node,
        critical=True
    )
    membership_urls = _combine_urls(basic.pac12_membership_urls)

    # Institution_Identification (under Pac-12 membership context)
    pac12_inst_leaf = evaluator.add_leaf(
        id="Institution_Identification_Pac12",
        desc="Identifies the institution that hired the coach",
        parent=pac12_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} was hired by {basic.hiring_institution}.",
        node=pac12_inst_leaf,
        sources=_combine_urls(basic.hiring_urls),
        additional_instruction="Confirm the hiring institution via the hiring announcement or official page."
    )

    # Pac12_Member_Verification (Parallel, Critical)
    pac12_ver_node = evaluator.add_parallel(
        id="Pac12_Member_Verification",
        desc="Confirms the institution is listed among the 2026 Pac-12 football-playing members",
        parent=pac12_node,
        critical=True
    )
    # Eight_Member_List_Check
    members_list_str = ", ".join(PAC12_2026_MEMBERS)
    eight_leaf = evaluator.add_leaf(
        id="Eight_Member_List_Check",
        desc="Verifies the institution appears in the official list of eight 2026 Pac-12 football members (Boise State, Colorado State, Fresno State, Oregon State, San Diego State, Texas State, Utah State, Washington State)",
        parent=pac12_ver_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The institution {basic.hiring_institution} is among the eight 2026 Pac-12 football members: {members_list_str}.",
        node=eight_leaf,
        sources=membership_urls,
        additional_instruction="Use the provided membership sources (conference announcements or credible news) to confirm inclusion."
    )
    # 2026_Start_Date_Confirmation
    start_leaf = evaluator.add_leaf(
        id="2026_Start_Date_Confirmation",
        desc="Confirms the institution's Pac-12 membership begins in 2026",
        parent=pac12_ver_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{basic.hiring_institution}'s Pac-12 football membership begins in 2026.",
        node=start_leaf,
        sources=membership_urls,
        additional_instruction="Verify that the source explicitly states the membership start is in 2026."
    )

    # Pac12_Membership_URL (existence)
    evaluator.add_custom_node(
        result=len(membership_urls) > 0,
        id="Pac12_Membership_URL",
        desc="Provides URL documentation for the institution's 2026 Pac-12 membership",
        parent=pac12_node,
        critical=True
    )

    # Hiring_Timeline (Sequential, Critical)
    hire_node = evaluator.add_sequential(
        id="Hiring_Timeline",
        desc="Confirms the coach was hired between September 2024 and January 2026",
        parent=node,
        critical=True
    )
    hiring_urls = _combine_urls(basic.hiring_urls)

    # Hiring_Date_Identification
    hire_date_leaf = evaluator.add_leaf(
        id="Hiring_Date_Identification",
        desc="Identifies the specific date or month when the coach was officially hired",
        parent=hire_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} was officially hired on {basic.hiring_date}.",
        node=hire_date_leaf,
        sources=hiring_urls,
        additional_instruction="Use the official announcement or credible news report to verify the stated hiring date."
    )

    # Timeline_Verification (Parallel, Critical)
    tv_node = evaluator.add_parallel(
        id="Timeline_Verification",
        desc="Verifies the hiring date falls within the September 2024 to January 2026 timeframe",
        parent=hire_node,
        critical=True
    )
    # Not_Before_September_2024
    nba_leaf = evaluator.add_leaf(
        id="Not_Before_September_2024",
        desc="Confirms the hiring occurred on or after September 1, 2024",
        parent=tv_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hiring date {basic.hiring_date} is on or after 2024-09-01.",
        node=nba_leaf,
        sources=hiring_urls,
        additional_instruction="Check the stated hiring date on the page. If only month/year is given (e.g., 'September 2024'), accept as on/after 2024-09-01 if consistent."
    )
    # Not_After_January_2026
    naf_leaf = evaluator.add_leaf(
        id="Not_After_January_2026",
        desc="Confirms the hiring occurred on or before January 31, 2026",
        parent=tv_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hiring date {basic.hiring_date} is on or before 2026-01-31.",
        node=naf_leaf,
        sources=hiring_urls,
        additional_instruction="Check the stated hiring date on the page. If only month/year is given (e.g., 'January 2026'), accept as within range if consistent."
    )

    # Hiring_Timeline_URL (existence)
    evaluator.add_custom_node(
        result=len(hiring_urls) > 0,
        id="Hiring_Timeline_URL",
        desc="Provides URL documentation for the hiring announcement and date",
        parent=hire_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_coach_info(),
        template_class=CoachExtraction,
        extraction_name="coach_extraction"
    )

    # Optional reference info (for transparency)
    evaluator.add_custom_info(
        info={"pac12_2026_expected_members": PAC12_2026_MEMBERS,
              "major_conferences": MAJOR_CONFERENCES},
        info_type="reference",
        info_name="reference_lists"
    )

    # Top-level grouping node mirroring the rubric task (set non-critical to allow partial credit)
    task_node = evaluator.add_parallel(
        id="Coach_Identification_Task",
        desc="Identifies a college football head coach hired between Sep 2024 and Jan 2026 who meets all specified career, performance, and institutional criteria",
        parent=root,
        critical=False
    )

    # Build subtrees
    await build_career_experience_verification(evaluator, task_node, extraction)
    await build_performance_record_verification(evaluator, task_node, extraction)
    await build_institutional_context_verification(evaluator, task_node, extraction)

    # Return evaluation summary
    return evaluator.get_summary()