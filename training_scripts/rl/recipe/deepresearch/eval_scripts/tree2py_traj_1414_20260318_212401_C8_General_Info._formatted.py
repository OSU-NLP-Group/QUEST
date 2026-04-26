import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "entertainment_achievements_2024_2026"
TASK_DESCRIPTION = """
Identify two American individuals from the entertainment industry (music, television, reality TV, or sports entertainment) who each achieved at least two major, publicly verifiable milestones between January 2024 and March 2026. For each person, at least one milestone must be either: (a) winning or achieving a top placement in a nationally televised competition (such as a reality TV show or dance competition), or (b) achieving a major music chart accomplishment (such as reaching #1 on a prominent national chart like the Billboard Hot 100). For each person and each of their achievements, provide: their full name, verification of their American nationality, the specific nature and title of the achievement, exact date(s), the platform/venue/chart, the specific outcome or result, and reference URLs to support each major claim.
"""

DATE_RANGE_TEXT = "between January 1, 2024 and March 31, 2026 (inclusive)"

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AchievementExtraction(BaseModel):
    nature_title: Optional[str] = None  # e.g., "Won Season 33 of Dancing with the Stars" or "Song X reached #1"
    date: Optional[str] = None          # free-form date as written in the answer (e.g., "March 15, 2025", "2024-11-20")
    platform_or_chart: Optional[str] = None  # e.g., "Dancing with the Stars (ABC)", "Billboard Hot 100"
    outcome: Optional[str] = None       # e.g., "Winner", "Runner-up", "#1 peak", "Top 3 finalist"
    urls: List[str] = Field(default_factory=list)  # reference URLs for this achievement


class PersonExtraction(BaseModel):
    full_name: Optional[str] = None
    nationality: Optional[str] = None               # as stated in the answer (e.g., "American", "U.S.")
    identity_urls: List[str] = Field(default_factory=list)  # URLs supporting identity/nationality
    achievement_1: Optional[AchievementExtraction] = None
    achievement_2: Optional[AchievementExtraction] = None


class TwoPeopleExtraction(BaseModel):
    person_1: Optional[PersonExtraction] = None
    person_2: Optional[PersonExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_people() -> str:
    return """
    Extract exactly two individuals (person_1 and person_2) from the entertainment industry (music, television, reality TV, or sports entertainment) as described in the answer, along with two achievements for each person.

    IMPORTANT:
    - Extract fields EXACTLY as presented in the answer text; do not invent or "fix" information.
    - If the answer lists more than two people, select the first two that appear.
    - For each person, if more than two achievements are listed, select the first two that appear in the answer (prefer those within Jan 2024–Mar 2026 when possible, but do not invent or alter dates).
    - If any field is missing in the answer, return null for that field (or an empty list for URLs).
    - For any URL fields, extract the actual URLs explicitly present in the answer (plain URL or markdown link). Do not fabricate URLs.

    Return a JSON object with this exact structure:
    {
      "person_1": {
        "full_name": str or null,
        "nationality": str or null,
        "identity_urls": [url, ...],  // may be empty
        "achievement_1": {
          "nature_title": str or null,
          "date": str or null,
          "platform_or_chart": str or null,
          "outcome": str or null,
          "urls": [url, ...]          // may be empty
        } or null,
        "achievement_2": {
          "nature_title": str or null,
          "date": str or null,
          "platform_or_chart": str or null,
          "outcome": str or null,
          "urls": [url, ...]          // may be empty
        } or null
      },
      "person_2": {
        "full_name": str or null,
        "nationality": str or null,
        "identity_urls": [url, ...],
        "achievement_1": { ... } or null,
        "achievement_2": { ... } or null
      }
    }

    Notes on interpretation:
    - "nature_title" should summarize the specific achievement (e.g., "Won Season 18 of The Voice", "Single 'XYZ' reached #1 on Billboard Hot 100").
    - "date" should be the exact date or date range as written in the answer (e.g., "Nov 20, 2024", "Week of March 1, 2025").
    - "platform_or_chart" can be a TV show/network (e.g., "The Voice (NBC)"), a competition, a venue, or a chart (e.g., "Billboard Hot 100").
    - "outcome" is the result (e.g., "Winner", "Runner-up", "Top 3 finalist", "#1", "Peaked at #1").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if _non_empty_str(u)]


def _merge_urls(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst:
            if _non_empty_str(u) and u not in seen:
                merged.append(u)
                seen.add(u)
    return merged


def _ach_brief(ach: Optional[AchievementExtraction]) -> str:
    if not ach:
        return "None"
    parts = []
    if _non_empty_str(ach.nature_title):
        parts.append(f"title: '{ach.nature_title}'")
    if _non_empty_str(ach.platform_or_chart):
        parts.append(f"platform/chart: '{ach.platform_or_chart}'")
    if _non_empty_str(ach.outcome):
        parts.append(f"outcome: '{ach.outcome}'")
    if _non_empty_str(ach.date):
        parts.append(f"date: '{ach.date}'")
    if not parts:
        return "None"
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_identity(
    evaluator: Evaluator,
    parent_node,
    person: Optional[PersonExtraction],
    person_idx: int
) -> None:
    """Build identity verification subtree for one person."""
    pid = f"person_{person_idx}"

    identity_node = evaluator.add_parallel(
        id=f"{pid}_identity",
        desc=f"Identity verification for {'first' if person_idx == 1 else 'second'} person",
        parent=parent_node,
        critical=True
    )

    name_value = person.full_name if person else None
    nationality_value = person.nationality if person else None
    identity_urls = _safe_urls(person.identity_urls if person else [])

    # Name provided (existence)
    evaluator.add_custom_node(
        result=_non_empty_str(name_value),
        id=f"{pid}_name",
        desc="Full name of the individual is provided",
        parent=identity_node,
        critical=True
    )

    # Identity URL provided (existence)
    evaluator.add_custom_node(
        result=len(identity_urls) > 0,
        id=f"{pid}_identity_url",
        desc="Reference URL provided supporting identity and nationality",
        parent=identity_node,
        critical=True
    )

    # Nationality verified with evidence
    nationality_leaf = evaluator.add_leaf(
        id=f"{pid}_nationality",
        desc="American nationality is verified with evidence",
        parent=identity_node,
        critical=True
    )

    name_for_claim = name_value or "the person"
    nat_claim = (
        f"{name_for_claim} is American (i.e., identified as an American national/citizen or described as "
        f"an 'American' [profession], or clearly indicated as U.S.-born/US citizen) according to the cited source(s)."
    )
    await evaluator.verify(
        claim=nat_claim,
        node=nationality_leaf,
        sources=identity_urls,
        additional_instruction=(
            "Accept clear textual evidence such as 'American singer', 'American television personality', "
            "'born in the United States' (and not exclusively elsewhere), or explicit U.S. citizenship. "
            "Minor wording variations are acceptable. If the page is unrelated or doesn't establish nationality, mark as not supported."
        )
    )


async def verify_achievement(
    evaluator: Evaluator,
    parent_node,
    person_name: Optional[str],
    ach: Optional[AchievementExtraction],
    person_idx: int,
    ach_idx: int
) -> None:
    """Build achievement verification subtree for one specific achievement."""
    pid = f"person_{person_idx}"
    aid = f"{pid}_achievement_{ach_idx}"

    ach_node = evaluator.add_parallel(
        id=aid,
        desc=f"{'First' if ach_idx == 1 else 'Second'} major achievement documented with complete details",
        parent=parent_node,
        critical=True
    )

    nature_val = ach.nature_title if ach else None
    date_val = ach.date if ach else None
    platform_val = ach.platform_or_chart if ach else None
    outcome_val = ach.outcome if ach else None
    ach_urls = _safe_urls(ach.urls if ach else [])

    # Specific nature/title provided (existence)
    evaluator.add_custom_node(
        result=_non_empty_str(nature_val),
        id=f"{pid}_ach{ach_idx}_nature",
        desc="Specific nature and title of the achievement provided",
        parent=ach_node,
        critical=True
    )

    # Exact date(s) provided (existence)
    evaluator.add_custom_node(
        result=_non_empty_str(date_val),
        id=f"{pid}_ach{ach_idx}_date",
        desc="Exact date(s) of achievement provided",
        parent=ach_node,
        critical=True
    )

    # Date falls within the timeframe (Jan 2024 – Mar 2026) [logical check]
    timeframe_leaf = evaluator.add_leaf(
        id=f"{pid}_ach{ach_idx}_timeframe",
        desc=f"Achievement date falls within January 2024 to March 2026",
        parent=ach_node,
        critical=True
    )
    tf_claim = (
        f"The provided date string '{date_val or ''}' falls {DATE_RANGE_TEXT}."
    )
    await evaluator.verify(
        claim=tf_claim,
        node=timeframe_leaf,
        additional_instruction=(
            "Judge the date purely by the string provided (do not search the web). "
            "Interpret common formats flexibly (e.g., 'March 2025', '2025-03-15', 'Week of Mar 1, 2025', 'Jan–Feb 2024'). "
            "Consider inclusive boundaries: 2024-01-01 to 2026-03-31. "
            "If a range is given, it qualifies if it occurs entirely within the window or overlaps such that the achievement date falls in the window. "
            "If the string is missing, ambiguous, or clearly out of range, mark as incorrect."
        )
    )

    # Platform/venue/chart specified (existence)
    evaluator.add_custom_node(
        result=_non_empty_str(platform_val),
        id=f"{pid}_ach{ach_idx}_platform",
        desc="Platform, venue, or chart specified",
        parent=ach_node,
        critical=True
    )

    # Specific outcome/result documented — verify against provided URLs
    outcome_leaf = evaluator.add_leaf(
        id=f"{pid}_ach{ach_idx}_outcome",
        desc="Specific outcome or result documented (e.g., placement, win status)",
        parent=ach_node,
        critical=True
    )
    oname = person_name or "the person"
    outcome_claim = (
        f"According to the cited source(s), on '{platform_val or ''}', {oname} achieved the following result: '{outcome_val or ''}' "
        f"for the achievement '{nature_val or ''}'."
    )
    await evaluator.verify(
        claim=outcome_claim,
        node=outcome_leaf,
        sources=ach_urls,
        additional_instruction=(
            "Focus on verifying the stated outcome/result on the specified platform/venue/chart. "
            "Allow minor wording differences (e.g., 'winner' vs. 'champion', 'No. 1' vs. '#1'). "
            "If the URL content is unrelated, inaccessible, or does not support the claimed outcome, mark as not supported."
        )
    )

    # Reference URL provided (existence)
    evaluator.add_custom_node(
        result=len(ach_urls) > 0,
        id=f"{pid}_ach{ach_idx}_url",
        desc="Reference URL provided supporting the achievement",
        parent=ach_node,
        critical=True
    )


async def verify_qualifying_requirement(
    evaluator: Evaluator,
    parent_node,
    person: Optional[PersonExtraction],
    person_idx: int
) -> None:
    """Verify that at least one of the two achievements qualifies as required."""
    pid = f"person_{person_idx}"

    qual_leaf = evaluator.add_leaf(
        id=f"{pid}_qualifying_achievement",
        desc=("At least one achievement is either a competition win/high placement in a nationally televised event "
              "OR a major chart accomplishment on a prominent national chart (e.g., Billboard Hot 100)"),
        parent=parent_node,
        critical=True
    )

    name_val = person.full_name if person else None
    a1 = person.achievement_1 if person else None
    a2 = person.achievement_2 if person else None
    srcs = _merge_urls(_safe_urls(a1.urls if a1 else []), _safe_urls(a2.urls if a2 else []))

    brief1 = _ach_brief(a1)
    brief2 = _ach_brief(a2)
    pname = name_val or "the person"

    qual_claim = (
        f"At least one of the following achievements by {pname} qualifies as either: "
        f"(a) a win or top placement in a nationally televised U.S. competition, or "
        f"(b) reaching #1 on a prominent U.S. national music chart (e.g., Billboard Hot 100 or Billboard 200).\n"
        f"Achievement 1: {brief1}\n"
        f"Achievement 2: {brief2}"
    )

    await evaluator.verify(
        claim=qual_claim,
        node=qual_leaf,
        sources=srcs,
        additional_instruction=(
            "Qualification rules:\n"
            "• Nationally televised competition examples include major U.S. TV shows: The Voice (NBC), American Idol (ABC), America's Got Talent (NBC), "
            "Dancing with the Stars (ABC), The Masked Singer (FOX), So You Think You Can Dance (FOX), etc. "
            "Winning or top final placements (winner/champion, runner-up, or top 3 finalist) qualify.\n"
            "• Major U.S. chart accomplishments include achieving #1 on Billboard Hot 100 (singles) or Billboard 200 (albums). "
            "Genre sub-charts or non-U.S. charts generally do NOT count as 'prominent national' unless the page explicitly positions them as such.\n"
            "Judge based on explicit evidence from the provided sources. It is sufficient if any one URL clearly supports a qualifying achievement."
        )
    )


async def verify_person(
    evaluator: Evaluator,
    root,
    person: Optional[PersonExtraction],
    person_idx: int
) -> None:
    """Build the verification subtree for one person (identity + two achievements + qualification)."""
    pid = f"person_{person_idx}"
    person_node = evaluator.add_parallel(
        id=pid,
        desc=f"{'First' if person_idx == 1 else 'Second'} American entertainment figure documented with at least two achievements",
        parent=root,
        critical=False  # Allow partial credit across two persons at root level
    )

    # Identity
    await verify_identity(evaluator, person_node, person, person_idx)

    # Achievements (two required)
    await verify_achievement(evaluator, person_node, person.full_name if person else None, person.achievement_1 if person else None, person_idx, 1)
    await verify_achievement(evaluator, person_node, person.full_name if person else None, person.achievement_2 if person else None, person_idx, 2)

    # Qualifying requirement (at least one achievement meets stricter condition)
    await verify_qualifying_requirement(evaluator, person_node, person, person_idx)


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
    Evaluate an answer for the 'two American entertainment figures with 2024–2026 achievements' task.
    """
    evaluator = Evaluator()
    # Note: We intentionally set root as non-critical to allow partial scoring if only one person is correct.
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

    # Record evaluation-time window info for transparency
    evaluator.add_custom_info(
        info={"accepted_time_window": "2024-01-01 to 2026-03-31 (inclusive)"},
        info_type="time_window",
        info_name="time_window_policy"
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_people(),
        template_class=TwoPeopleExtraction,
        extraction_name="people_extraction"
    )

    # Build verification tree for two persons (always create both branches)
    await verify_person(evaluator, root, extracted.person_1 if extracted else None, 1)
    await verify_person(evaluator, root, extracted.person_2 if extracted else None, 2)

    return evaluator.get_summary()