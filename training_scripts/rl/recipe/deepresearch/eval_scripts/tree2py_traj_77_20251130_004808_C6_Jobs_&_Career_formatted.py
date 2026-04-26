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
TASK_ID = "big_ten_coach_2025_criteria"
TASK_DESCRIPTION = """
Identify a current head football coach at a Big Ten Conference school (as of the 2025 season) who meets all of the following career qualifications: 
(1) previously served as a defensive coordinator at a Southeastern Conference (SEC) member institution, 
(2) previously served as a head coach at a non-Power 5 FBS program, 
(3) has earned conference Coach of the Year recognition during their head coaching career, and 
(4) has led a team to a conference championship game appearance as a head coach. 
Provide the coach's name, their current institution, and documentation with reference URLs for each of the four qualification criteria.
"""
CURRENT_SEASON = 2025


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SECDefenseCoordInfo(BaseModel):
    description: Optional[str] = None
    school: Optional[str] = None
    role_title: Optional[str] = None
    years: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class NonP5HeadCoachInfo(BaseModel):
    description: Optional[str] = None
    program: Optional[str] = None
    conference: Optional[str] = None
    years: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CoachOfYearInfo(BaseModel):
    description: Optional[str] = None
    conference: Optional[str] = None
    year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ChampGameInfo(BaseModel):
    description: Optional[str] = None
    conference: Optional[str] = None
    season: Optional[str] = None
    game_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CoachProfileExtraction(BaseModel):
    coach_name: Optional[str] = None
    current_institution: Optional[str] = None

    current_role_description: Optional[str] = None
    current_institution_urls: List[str] = Field(default_factory=list)
    current_big_ten_urls: List[str] = Field(default_factory=list)

    sec_defensive_coordinator: Optional[SECDefenseCoordInfo] = None
    non_power5_head_coach: Optional[NonP5HeadCoachInfo] = None
    coach_of_year: Optional[CoachOfYearInfo] = None
    conference_championship_game: Optional[ChampGameInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_profile() -> str:
    return f"""
    Extract structured information for exactly ONE coach (the first coach if multiple are mentioned) from the answer. 
    Return a JSON object matching the schema below. Extract only information explicitly present in the answer.

    Schema:
    - coach_name: string or null
    - current_institution: string or null
    - current_role_description: string or null (e.g., "Head coach at X since 2024")
    - current_institution_urls: array of URL strings (any links in the answer that document the coach's current role at the institution)
    - current_big_ten_urls: array of URL strings (any links in the answer that indicate the institution's Big Ten membership or the coach's Big Ten head-coach status)

    - sec_defensive_coordinator: object or null
        - description: string or null (e.g., "Defensive Coordinator at Florida (2017-2019)")
        - school: string or null (SEC school name)
        - role_title: string or null (e.g., "Defensive Coordinator" or "Co-Defensive Coordinator")
        - years: string or null (e.g., "2018-2019")
        - urls: array of URL strings (links documenting this SEC DC role)

    - non_power5_head_coach: object or null
        - description: string or null (e.g., "Head Coach at Toledo in MAC")
        - program: string or null (school/program name)
        - conference: string or null (conference name such as AAC, C-USA, MAC, Mountain West, Sun Belt)
        - years: string or null
        - urls: array of URL strings (links documenting this head-coach role at a non-Power 5 FBS program)

    - coach_of_year: object or null
        - description: string or null (e.g., "AAC Coach of the Year in 2020")
        - conference: string or null
        - year: string or null
        - urls: array of URL strings (links documenting the conference Coach of the Year award)

    - conference_championship_game: object or null
        - description: string or null (e.g., "Led team to MAC Championship Game in 2019")
        - conference: string or null
        - season: string or null
        - game_name: string or null (e.g., "AAC Championship Game")
        - urls: array of URL strings (links documenting the appearance in a conference championship game as head coach)

    Rules:
    - Extract only URLs that appear in the answer (including markdown links).
    - For any missing information, return null for strings or empty array for URLs.
    - Do not invent or infer information beyond what the answer explicitly states.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _combine_sources(*lists: List[str]) -> List[str]:
    out: List[str] = []
    for l in lists:
        out.extend([u for u in l if _non_empty(u)])
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_current_big_ten_head_coach(
    evaluator: Evaluator,
    parent_node,
    coach: CoachProfileExtraction,
) -> None:
    """
    Create and verify the 'CurrentBigTenHeadCoach2025' leaf node.
    """
    node = evaluator.add_leaf(
        id="current_big_ten_head_coach_2025",
        desc="The identified person is a current head football coach at a Big Ten Conference member institution as of the 2025 season.",
        parent=parent_node,
        critical=True,
    )

    coach_name = coach.coach_name or ""
    institution = coach.current_institution or ""
    claim = (
        f"As of the {CURRENT_SEASON} season, {coach_name} is the head football coach at {institution}, "
        f"and {institution} is a Big Ten Conference member institution."
    )

    sources = _combine_sources(coach.current_institution_urls, coach.current_big_ten_urls)

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources if sources else None,
        additional_instruction=(
            "Confirm BOTH parts using the provided sources: "
            "(1) the person is the head football coach at the specified institution, and "
            f"(2) the institution competes in the Big Ten Conference for the {CURRENT_SEASON} season. "
            "Allow reasonable preseason/announcement timing if it clearly pertains to the 2025 season. "
            "If a single page does not state both facts, the claim is not fully supported by that page."
        ),
    )


async def verify_criterion_block(
    evaluator: Evaluator,
    parent_node,
    block_id: str,
    block_desc: str,
    reference_desc: str,
    criterion_info: Optional[BaseModel],
    make_claim_fn,
    additional_instruction: str,
    coach_name: str,
) -> None:
    """
    Generic builder for a qualification block with:
      - ReferenceURLProvided: existence check (critical)
      - CriterionMet: verification by URLs (critical)
    """
    # Add the parallel block node (critical)
    block_node = evaluator.add_parallel(
        id=block_id,
        desc=block_desc,
        parent=parent_node,
        critical=True,
    )

    urls: List[str] = []
    if criterion_info and hasattr(criterion_info, "urls") and isinstance(criterion_info.urls, list):
        urls = criterion_info.urls

    # Reference URL existence check (critical)
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id=f"{block_id}_reference_url_provided",
        desc=reference_desc,
        parent=block_node,
        critical=True,
    )

    # CriterionMet verification (critical)
    criterion_node = evaluator.add_leaf(
        id=f"{block_id}_criterion_met",
        desc="Criterion is supported by cited sources",
        parent=block_node,
        critical=True,
    )

    claim_text = make_claim_fn(criterion_info, coach_name)

    await evaluator.verify(
        claim=claim_text,
        node=criterion_node,
        sources=urls if urls else None,
        additional_instruction=additional_instruction,
    )


# Claim builders
def make_claim_sec_dc(info: Optional[SECDefenseCoordInfo], coach_name: str) -> str:
    if info and _non_empty(info.school):
        role = info.role_title or "defensive coordinator"
        return (
            f"{coach_name} previously served as {role} at {info.school}, "
            "which is a member of the Southeastern Conference (SEC)."
        )
    return f"{coach_name} previously served as a defensive coordinator at an SEC member institution."


def make_claim_non_p5_head(info: Optional[NonP5HeadCoachInfo], coach_name: str) -> str:
    if info and _non_empty(info.program):
        conf_part = f" in the {info.conference}" if _non_empty(info.conference) else ""
        return (
            f"{coach_name} previously served as the head football coach at {info.program}{conf_part}, "
            "which is a non-Power 5 FBS (Group of Five) program."
        )
    return f"{coach_name} previously served as a head coach at a non-Power 5 FBS (Group of Five) program."


def make_claim_coy(info: Optional[CoachOfYearInfo], coach_name: str) -> str:
    if info and (_non_empty(info.conference) or _non_empty(info.year)):
        conf = info.conference or "a conference"
        year = info.year or "a given year"
        return f"As a head coach, {coach_name} earned {conf} Coach of the Year honors in {year}."
    return f"As a head coach, {coach_name} earned conference Coach of the Year recognition."


def make_claim_champ_game(info: Optional[ChampGameInfo], coach_name: str) -> str:
    parts = []
    if info and _non_empty(info.conference):
        parts.append(f"{info.conference} conference")
    else:
        parts.append("a conference")

    game = info.game_name if info and _non_empty(info.game_name) else "championship game"
    season = f" in {info.season}" if info and _non_empty(info.season) else ""
    return f"As head coach, {coach_name} led a team to a {parts[0]} {game}{season} appearance."


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
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for the Big Ten coach qualification task.
    """
    # Initialize evaluator
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

    # Create a critical top-level task node to mirror the rubric's critical Root
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify one current (2025 season) Big Ten head football coach who meets all listed career qualifications; provide coach name, current institution, and a reference URL for each qualification criterion.",
        parent=root,
        critical=True,
    )

    # Extract the structured profile
    coach_profile = await evaluator.extract(
        prompt=prompt_extract_coach_profile(),
        template_class=CoachProfileExtraction,
        extraction_name="coach_profile",
    )

    # 1) AnswerIncludesCoachName (critical existence check)
    evaluator.add_custom_node(
        result=_non_empty(coach_profile.coach_name),
        id="answer_includes_coach_name",
        desc="Provides the coach's name.",
        parent=task_root,
        critical=True,
    )

    # 2) AnswerIncludesCurrentInstitution (critical existence check)
    evaluator.add_custom_node(
        result=_non_empty(coach_profile.current_institution),
        id="answer_includes_current_institution",
        desc="Provides the coach's current institution.",
        parent=task_root,
        critical=True,
    )

    # 3) CurrentBigTenHeadCoach2025 (critical verification)
    await verify_current_big_ten_head_coach(
        evaluator=evaluator,
        parent_node=task_root,
        coach=coach_profile,
    )

    # 4) SECDefensiveCoordinator block
    await verify_criterion_block(
        evaluator=evaluator,
        parent_node=task_root,
        block_id="sec_defensive_coordinator",
        block_desc="SEC defensive coordinator qualification (criterion + citation).",
        reference_desc="Provides at least one reference URL documenting the SEC defensive coordinator criterion.",
        criterion_info=coach_profile.sec_defensive_coordinator,
        make_claim_fn=make_claim_sec_dc,
        additional_instruction=(
            "Verify that the coach previously served as a defensive coordinator at a school that is a member of the SEC. "
            "Titles such as 'Defensive Coordinator' or 'Co-Defensive Coordinator' are acceptable. "
            "Position-coach roles (e.g., linebackers coach) or analyst roles are not sufficient."
        ),
        coach_name=coach_profile.coach_name or "",
    )

    # 5) Non-Power 5 FBS Head Coach block
    await verify_criterion_block(
        evaluator=evaluator,
        parent_node=task_root,
        block_id="non_power5_fbs_head_coach",
        block_desc="Non-Power 5 FBS head coach qualification (criterion + citation).",
        reference_desc="Provides at least one reference URL documenting the non-Power 5 FBS head coach criterion.",
        criterion_info=coach_profile.non_power5_head_coach,
        make_claim_fn=make_claim_non_p5_head,
        additional_instruction=(
            "Verify that the coach served as head coach at a non-Power 5 FBS program (Group of Five: AAC, C-USA, MAC, Mountain West, Sun Belt). "
            "Power 5 conferences (e.g., Big Ten, SEC, ACC, Big 12, Pac-12) do NOT satisfy this criterion. "
            "FCS or non-FBS programs do not satisfy this criterion."
        ),
        coach_name=coach_profile.coach_name or "",
    )

    # 6) Conference Coach of the Year block
    await verify_criterion_block(
        evaluator=evaluator,
        parent_node=task_root,
        block_id="conference_coach_of_year",
        block_desc="Conference Coach of the Year qualification (criterion + citation).",
        reference_desc="Provides at least one reference URL documenting the Coach of the Year criterion.",
        criterion_info=coach_profile.coach_of_year,
        make_claim_fn=make_claim_coy,
        additional_instruction=(
            "Verify that the coach earned a conference-level 'Coach of the Year' honor during their head-coaching career. "
            "National coach-of-the-year awards do not qualify unless explicitly conference-level. "
            "Co-Coach of the Year is acceptable."
        ),
        coach_name=coach_profile.coach_name or "",
    )

    # 7) Conference Championship Game Appearance block
    await verify_criterion_block(
        evaluator=evaluator,
        parent_node=task_root,
        block_id="conference_championship_game_appearance",
        block_desc="Conference championship game appearance qualification (criterion + citation).",
        reference_desc="Provides at least one reference URL documenting the conference championship game appearance criterion.",
        criterion_info=coach_profile.conference_championship_game,
        make_claim_fn=make_claim_champ_game,
        additional_instruction=(
            "Verify that, while serving as head coach, the coach led a team to a conference championship game appearance "
            "(e.g., AAC Championship Game, C-USA Championship Game, MAC Championship Game, Mountain West Championship, Sun Belt Championship). "
            "Winning is not required; an appearance suffices."
        ),
        coach_name=coach_profile.coach_name or "",
    )

    # Optionally record meta-info
    evaluator.add_custom_info(
        {"season": CURRENT_SEASON},
        info_type="season_context",
        info_name="season_context",
    )

    return evaluator.get_summary()