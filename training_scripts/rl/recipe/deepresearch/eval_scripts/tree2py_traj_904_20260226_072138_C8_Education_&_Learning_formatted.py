import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ct_independent_coach_2022_2025"
TASK_DESCRIPTION = """
Identify the college football head coach who meets ALL of the following criteria:

1. Current/Recent Institution (2022-2025):
   - Served as head coach at an NCAA Division I FBS institution located in Connecticut
   - Tenure lasted exactly 4 seasons (2022-2025)
   - The institution competed as a football independent (not in a conference) during this period

2. Coaching Achievements at This Institution:
   - Led the program to its first 9-win season in at least 15 years
   - Won a bowl game, which was the program's first bowl victory since at least 2010
   - Significantly improved the program's performance from previous years

3. Prior Head Coaching Experience:
   - Previously served as head coach at UCLA from 2012-2017
   - Achieved a winning overall record at UCLA (specifically 46-30)
   - Had a gap of several years between the UCLA position and the Connecticut position

4. 2025 Career Move:
   - Left the Connecticut institution in November 2025
   - Accepted a head coaching position at Colorado State University
   - Was formally introduced at Colorado State in December 2025

5. Personal Background:
   - Has family connections to professional football coaching (parent was an NFL head coach)

Provide the coach's full name, verify that all criteria are met with specific evidence, and include supporting URL references for each major criterion.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConnecticutTenureInfo(BaseModel):
    institution: Optional[str] = None
    tenure_years: Optional[str] = None  # e.g., "2022–2025" or "2022-2025"
    seasons_count: Optional[str] = None  # e.g., "4" or "four"
    fbs_status: Optional[str] = None  # e.g., "NCAA Division I FBS"
    independent_status: Optional[str] = None  # e.g., "Independent"
    urls: List[str] = Field(default_factory=list)


class AchievementsInfo(BaseModel):
    nine_win_year: Optional[str] = None
    nine_win_phrase: Optional[str] = None  # e.g., "first 9-win season in 15+ years"
    bowl_win_year: Optional[str] = None
    bowl_win_phrase: Optional[str] = None  # e.g., "first bowl victory since 2010"
    improvement_desc: Optional[str] = None  # objective improvement statement
    urls: List[str] = Field(default_factory=list)


class UCLAInfo(BaseModel):
    years: Optional[str] = None  # e.g., "2012–2017"
    record: Optional[str] = None  # e.g., "46–30"
    gap_desc: Optional[str] = None  # e.g., "gap of several years before taking CT job in 2022"
    urls: List[str] = Field(default_factory=list)


class CareerMove2025Info(BaseModel):
    left_date: Optional[str] = None  # e.g., "Nov 2025"
    accepted_date: Optional[str] = None  # e.g., "Nov 2025"
    introduced_date: Optional[str] = None  # e.g., "Dec 2025"
    urls: List[str] = Field(default_factory=list)


class PersonalBackgroundInfo(BaseModel):
    parent_name: Optional[str] = None
    parent_role_desc: Optional[str] = None  # e.g., "father was an NFL head coach"
    urls: List[str] = Field(default_factory=list)


class CoachExtraction(BaseModel):
    coach_name: Optional[str] = None
    ct_tenure: Optional[ConnecticutTenureInfo] = None
    achievements: Optional[AchievementsInfo] = None
    ucla: Optional[UCLAInfo] = None
    career2025: Optional[CareerMove2025Info] = None
    background: Optional[PersonalBackgroundInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_info() -> str:
    return """
Extract the following structured information from the answer. Return JSON that strictly matches the specified schema. 
Do not invent any information; only use what appears in the answer text. Extract all URLs that the answer cites as evidence.

Fields:
- coach_name: The full name of the coach identified in the answer.

- ct_tenure:
  - institution: The name of the NCAA Division I FBS institution in Connecticut where the coach served from 2022–2025.
  - tenure_years: The years of tenure exactly as stated (e.g., "2022–2025" or "2022-2025").
  - seasons_count: The number of seasons stated (e.g., "4" or the word "four" if used).
  - fbs_status: Any mention confirming NCAA Division I FBS status.
  - independent_status: Any mention confirming football independent status (no conference).
  - urls: A list of URLs cited in the answer that support these tenure/location/status claims.

- achievements:
  - nine_win_year: The year of the 9-win season, if stated.
  - nine_win_phrase: The phrasing that indicates it was the first 9-win season in at least 15 years.
  - bowl_win_year: The year of the bowl win, if stated.
  - bowl_win_phrase: The phrasing that indicates it was the first bowl victory since at least 2010.
  - improvement_desc: Any explicit comparative statement that performance significantly improved versus prior seasons (e.g., improved record, postseason milestone).
  - urls: A list of URLs cited in the answer that support these achievement claims.

- ucla:
  - years: The UCLA head-coaching years (e.g., "2012–2017").
  - record: The aggregate record at UCLA (should be "46–30" if stated).
  - gap_desc: A description noting a multi-year gap between UCLA (ending 2017) and the Connecticut position (beginning 2022).
  - urls: A list of URLs cited in the answer that support the UCLA tenure/record/gap.

- career2025:
  - left_date: A phrasing indicating the coach left the Connecticut institution in November 2025.
  - accepted_date: A phrasing indicating the coach accepted the Colorado State head-coaching job in November 2025.
  - introduced_date: A phrasing indicating the coach was formally introduced at Colorado State in December 2025.
  - urls: A list of URLs cited in the answer that support the 2025 departure/acceptance/intro timeline.

- background:
  - parent_name: The name of the coach's parent who was an NFL head coach, if provided.
  - parent_role_desc: A phrasing that clearly states the parent was an NFL head coach.
  - urls: A list of URLs cited in the answer that support the parent NFL head-coach claim.

URL extraction rules:
- Extract only URLs explicitly present in the answer text (including plain URLs or markdown links).
- Do not infer or invent any URLs.
- Include full URLs; if protocol is missing, prepend "http://".
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _combine_urls(*lists: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    for l in lists:
        if l:
            urls.extend([u for u in l if isinstance(u, str) and u.strip()])
    # Optionally deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level criteria are independent checks
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

    # Extract structured info
    extracted: CoachExtraction = await evaluator.extract(
        prompt=prompt_extract_coach_info(),
        template_class=CoachExtraction,
        extraction_name="coach_extraction"
    )

    # Build a critical top-level node to enforce all criteria
    coach_root = evaluator.add_parallel(
        id="Coach_Identification",
        desc="Identify the college football head coach who satisfies all criteria in the proposed question and provide URL evidence for each major criterion.",
        parent=root,
        critical=True
    )

    # ------------------------ Coach_Full_Name (Critical) ------------------ #
    coach_name = (extracted.coach_name or "").strip()
    evaluator.add_custom_node(
        result=bool(coach_name),
        id="Coach_Full_Name",
        desc="Provide the coach's full name.",
        parent=coach_root,
        critical=True
    )

    # ----------------- Connecticut_FBS_Independent_Tenure ----------------- #
    ct_node = evaluator.add_parallel(
        id="Connecticut_FBS_Independent_Tenure_2022_2025",
        desc="Verify the coach served as head coach at an NCAA Division I FBS institution in Connecticut that played as an independent, for exactly four seasons (2022–2025).",
        parent=coach_root,
        critical=True
    )

    ct = extracted.ct_tenure or ConnecticutTenureInfo()
    ct_urls = _safe_list(ct.urls)
    institution = (ct.institution or "the institution").strip()
    tenure_years = (ct.tenure_years or "2022–2025").strip()

    # Institution in Connecticut
    ct_loc_leaf = evaluator.add_leaf(
        id="Institution_In_Connecticut",
        desc="The institution is located in Connecticut.",
        parent=ct_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{institution} is located in the U.S. state of Connecticut.",
        node=ct_loc_leaf,
        sources=ct_urls,
        additional_instruction="Verify location. Accept reasonable variants (e.g., 'Storrs, Connecticut'). The claim is supported if the institution is clearly identified as being in Connecticut."
    )

    # NCAA Division I FBS
    ct_fbs_leaf = evaluator.add_leaf(
        id="Institution_Is_NCAA_Division_I_FBS",
        desc="The institution is an NCAA Division I FBS program.",
        parent=ct_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{institution} competes in NCAA Division I FBS football.",
        node=ct_fbs_leaf,
        sources=ct_urls,
        additional_instruction="Verify that the football program is NCAA Division I Football Bowl Subdivision (FBS). Distinguish from FCS."
    )

    # Independent status during tenure
    ct_ind_leaf = evaluator.add_leaf(
        id="Independent_Status_During_Tenure",
        desc="The institution competed as a football independent (not in a conference) during the coach's tenure.",
        parent=ct_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"During {tenure_years}, {institution} competed as an FBS independent (not affiliated with a football conference).",
        node=ct_ind_leaf,
        sources=ct_urls,
        additional_instruction="Confirm that for the specified tenure window, the football program is listed as 'Independent' (no conference)."
    )

    # Tenure exactly four seasons 2022–2025
    ct_tenure_leaf = evaluator.add_leaf(
        id="Tenure_Exactly_Four_Seasons_2022_2025",
        desc="The coach served exactly four seasons spanning 2022–2025.",
        parent=ct_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} served as the head coach at {institution} from 2022 through 2025, exactly four seasons.",
        node=ct_tenure_leaf,
        sources=ct_urls,
        additional_instruction="Check tenure dates and count seasons inclusively: 2022, 2023, 2024, 2025."
    )

    # --------------- Coaching_Achievements_At_Connecticut ----------------- #
    ach_node = evaluator.add_parallel(
        id="Coaching_Achievements_At_Connecticut_Institution",
        desc="Verify the specified on-field achievements at the Connecticut institution.",
        parent=coach_root,
        critical=True
    )

    ach = extracted.achievements or AchievementsInfo()
    ach_urls = _safe_list(ach.urls)

    # First 9-win season in at least 15 years
    nine_leaf = evaluator.add_leaf(
        id="First_9_Win_Season_In_At_Least_15_Years",
        desc="Led the program to its first 9-win season in at least 15 years.",
        parent=ach_node,
        critical=True
    )
    nine_year = (ach.nine_win_year or "").strip()
    nine_phrase = (ach.nine_win_phrase or "its first 9-win season in at least 15 years").strip()
    await evaluator.verify(
        claim=f"Under head coach {coach_name}, {institution} achieved a 9-win season{(' in ' + nine_year) if nine_year else ''}, {nine_phrase}.",
        node=nine_leaf,
        sources=ach_urls,
        additional_instruction="Confirm both: (1) a 9-win season occurred under this coach and (2) it was the first in at least ~15 years (allow small phrasing variants like 'first since YEAR')."
    )

    # First bowl win since at least 2010
    bowl_leaf = evaluator.add_leaf(
        id="First_Bowl_Win_Since_At_Least_2010",
        desc="Won a bowl game that was the program’s first bowl victory since at least 2010.",
        parent=ach_node,
        critical=True
    )
    bowl_year = (ach.bowl_win_year or "").strip()
    bowl_phrase = (ach.bowl_win_phrase or "its first bowl victory since at least 2010").strip()
    await evaluator.verify(
        claim=f"Under head coach {coach_name}, {institution} won a bowl game{(' in ' + bowl_year) if bowl_year else ''}, {bowl_phrase}.",
        node=bowl_leaf,
        sources=ach_urls,
        additional_instruction="Verify that a bowl victory occurred under this coach and that it was the first since at least 2010."
    )

    # Improved performance from previous years
    improve_leaf = evaluator.add_leaf(
        id="Improved_Performance_From_Previous_Years",
        desc="Provide objective evidence that the program’s performance improved relative to previous years (e.g., improved win-loss record and/or postseason attainment compared to pre-tenure seasons), consistent with the claim of significant improvement.",
        parent=ach_node,
        critical=True
    )
    improve_desc = (ach.improvement_desc or "the program's performance significantly improved compared to previous years").strip()
    await evaluator.verify(
        claim=f"Under {coach_name}, {institution} improved significantly compared to prior seasons (e.g., record/postseason): {improve_desc}.",
        node=improve_leaf,
        sources=ach_urls,
        additional_instruction="Look for objective markers (more wins, bowl eligibility/win, rankings) versus pre-tenure years."
    )

    # ---------------------- Prior_UCLA_Head_Coaching ---------------------- #
    ucla_node = evaluator.add_parallel(
        id="Prior_UCLA_Head_Coaching",
        desc="Verify the coach's prior UCLA head-coaching experience and related constraints.",
        parent=coach_root,
        critical=True
    )

    ucla = extracted.ucla or UCLAInfo()
    ucla_urls = _safe_list(ucla.urls)

    ucla_years_leaf = evaluator.add_leaf(
        id="UCLA_Head_Coach_2012_2017",
        desc="Previously served as head coach at UCLA from 2012–2017.",
        parent=ucla_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} served as UCLA's head football coach from 2012 to 2017.",
        node=ucla_years_leaf,
        sources=ucla_urls,
        additional_instruction="Confirm the coach's tenure years at UCLA are 2012–2017."
    )

    ucla_record_leaf = evaluator.add_leaf(
        id="UCLA_Record_46_30",
        desc="Achieved a 46–30 overall record at UCLA.",
        parent=ucla_node,
        critical=True
    )
    record_str = (ucla.record or "46–30").strip()
    await evaluator.verify(
        claim=f"{coach_name}'s overall record at UCLA was {record_str}.",
        node=ucla_record_leaf,
        sources=ucla_urls,
        additional_instruction="Verify the aggregate record (accept minor formatting variants like 46-30 vs 46–30)."
    )

    ucla_gap_leaf = evaluator.add_leaf(
        id="Gap_Several_Years_Between_UCLA_And_Connecticut",
        desc="Had a gap of several years between the UCLA position (ending 2017) and the Connecticut position (beginning 2022).",
        parent=ucla_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"After leaving UCLA in 2017, {coach_name} did not serve as a head coach again until taking the {institution} position in 2022, representing a multi-year gap.",
        node=ucla_gap_leaf,
        sources=_combine_urls(ucla_urls, ct_urls),
        additional_instruction="Confirm that there was no head-coaching post immediately after 2017 and that the next head-coaching role began in 2022 (a gap of several years)."
    )

    # -------------------------- Career_Move_2025 -------------------------- #
    career_node = evaluator.add_parallel(
        id="Career_Move_2025",
        desc="Verify the specified departure and hiring timeline in 2025.",
        parent=coach_root,
        critical=True
    )

    car = extracted.career2025 or CareerMove2025Info()
    car_urls = _safe_list(car.urls)

    left_leaf = evaluator.add_leaf(
        id="Left_Connecticut_Institution_Nov_2025",
        desc="Left the Connecticut institution in November 2025.",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} left {institution} in November 2025.",
        node=left_leaf,
        sources=car_urls,
        additional_instruction="Verify departure timing was in November 2025."
    )

    accepted_leaf = evaluator.add_leaf(
        id="Accepted_Colorado_State_Position_Nov_2025",
        desc="Accepted a head coaching position at Colorado State University in November 2025.",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In November 2025, {coach_name} accepted the head-coaching position at Colorado State University.",
        node=accepted_leaf,
        sources=car_urls,
        additional_instruction="Verify acceptance/hire date reported in November 2025 for Colorado State."
    )

    introduced_leaf = evaluator.add_leaf(
        id="Formally_Introduced_Dec_2025",
        desc="Was formally introduced at Colorado State in December 2025.",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{coach_name} was formally introduced at Colorado State in December 2025.",
        node=introduced_leaf,
        sources=car_urls,
        additional_instruction="Verify the formal introductory event took place in December 2025."
    )

    # -------------------------- Personal_Background ----------------------- #
    bg_node = evaluator.add_parallel(
        id="Personal_Background",
        desc="Verify the required family connection to professional football coaching.",
        parent=coach_root,
        critical=True
    )

    bg = extracted.background or PersonalBackgroundInfo()
    bg_urls = _safe_list(bg.urls)
    parent_name = (bg.parent_name or "the coach's parent").strip()
    parent_desc = (bg.parent_role_desc or "served as an NFL head coach").strip()

    parent_leaf = evaluator.add_leaf(
        id="Parent_Was_NFL_Head_Coach",
        desc="Has a parent who was an NFL head coach.",
        parent=bg_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{parent_name} {parent_desc}.",
        node=parent_leaf,
        sources=bg_urls,
        additional_instruction="Confirm that the coach's parent held an NFL head-coach position at some point."
    )

    # --------------------- Supporting_URL_References ---------------------- #
    urls_node = evaluator.add_parallel(
        id="Supporting_URL_References",
        desc="Include supporting URL references for each major criterion (at least one relevant URL per major criterion).",
        parent=coach_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(ct_urls) > 0,
        id="URL_For_Connecticut_Tenure_Criterion",
        desc="Provide at least one URL supporting the Connecticut FBS independent tenure criterion.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ach_urls) > 0,
        id="URL_For_Coaching_Achievements_Criterion",
        desc="Provide at least one URL supporting the coaching achievements criterion (9-win season / bowl win / improvement claim).",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ucla_urls) > 0,
        id="URL_For_Prior_UCLA_Criterion",
        desc="Provide at least one URL supporting the prior UCLA head-coaching criterion (years and record).",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(car_urls) > 0,
        id="URL_For_Career_Move_2025_Criterion",
        desc="Provide at least one URL supporting the 2025 career move timeline criterion.",
        parent=urls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(bg_urls) > 0,
        id="URL_For_Parent_NFL_Head_Coach_Criterion",
        desc="Provide at least one URL supporting the parent-was-an-NFL-head-coach criterion.",
        parent=urls_node,
        critical=True
    )

    # Return evaluation summary
    return evaluator.get_summary()