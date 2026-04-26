import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "emmys_2025_streaming_top_winners"
TASK_DESCRIPTION = """
Identify three streaming series that won the top awards in their respective categories at the 77th Primetime Emmy Awards (September 14, 2025): Outstanding Drama Series, Outstanding Comedy Series, and Outstanding Limited or Anthology Series. Each series must be from a different major streaming platform (Apple TV+, HBO Max, or Netflix).

For each series, provide:

For the Outstanding Drama Series winner:
- Series title and streaming platform
- Season number that won
- The actor who won Outstanding Lead Actor in a Drama Series for this show
- The actress who won Outstanding Supporting Actress in a Drama Series for this show
- Number of episodes in the winning season
- At least one reference URL

For the Outstanding Comedy Series winner:
- Series title and streaming platform
- Season number that won
- The actor who won Outstanding Lead Actor in a Comedy Series for this show
- The Emmy record this series set at the 2025 ceremony (specifically, it must be identified as the most-winning freshman comedy in Emmy history)
- The total number of Emmy wins it achieved
- Number of episodes in the winning season
- At least one reference URL

For the Outstanding Limited or Anthology Series winner:
- Series title and streaming platform
- The actor who won Outstanding Lead Actor in a Limited or Anthology Series for this show
- The actor who won Outstanding Supporting Actor in a Limited or Anthology Series for this show (must confirm he was 15 years old and became the youngest male Emmy winner in history)
- Confirmation that the series won at least 8 Emmy awards total
- Number of episodes in the series
- At least one reference URL
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class DramaSeriesInfo(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None
    season_number: Optional[str] = None
    lead_actor: Optional[str] = None  # Outstanding Lead Actor in a Drama Series (for this show)
    supporting_actress: Optional[str] = None  # Outstanding Supporting Actress in a Drama Series (for this show)
    episode_count: Optional[str] = None  # Number of episodes in the winning season
    urls: List[str] = Field(default_factory=list)


class ComedySeriesInfo(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None
    season_number: Optional[str] = None
    lead_actor: Optional[str] = None  # Outstanding Lead Actor in a Comedy Series (for this show)
    emmy_record: Optional[str] = None  # should indicate "most-winning freshman comedy in Emmy history"
    total_wins: Optional[str] = None  # should be "13"
    episode_count: Optional[str] = None  # Number of episodes in the winning season
    urls: List[str] = Field(default_factory=list)


class LimitedSeriesInfo(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None
    lead_actor: Optional[str] = None  # Outstanding Lead Actor in a Limited/Anthology (for this show)
    supporting_actor: Optional[str] = None  # Outstanding Supporting Actor in a Limited/Anthology (for this show)
    total_wins: Optional[str] = None  # should be >= 8
    episode_count: Optional[str] = None  # Number of episodes in the limited series
    urls: List[str] = Field(default_factory=list)


class WinnersExtraction(BaseModel):
    drama: Optional[DramaSeriesInfo] = None
    comedy: Optional[ComedySeriesInfo] = None
    limited: Optional[LimitedSeriesInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_winners() -> str:
    return """
    Extract the three Emmy-winning series and all requested fields from the provided answer.

    You must extract three sections:
    1) drama: The series that won Outstanding Drama Series at the 77th Primetime Emmy Awards (2025).
       - title
       - platform
       - season_number (the season that won)
       - lead_actor (the person who won Outstanding Lead Actor in a Drama Series for this show)
       - supporting_actress (the person who won Outstanding Supporting Actress in a Drama Series for this show)
       - episode_count (number of episodes in the winning season)
       - urls (array of at least one URL referenced for this show)

    2) comedy: The series that won Outstanding Comedy Series at the 77th Primetime Emmy Awards (2025).
       - title
       - platform
       - season_number (the season that won)
       - lead_actor (the person who won Outstanding Lead Actor in a Comedy Series for this show)
       - emmy_record (should explicitly mention "most-winning freshman comedy in Emmy history" or a very close paraphrase)
       - total_wins (the total number of Emmy wins at the 2025 ceremony; expected 13)
       - episode_count (number of episodes in the winning season)
       - urls (array of at least one URL referenced for this show)

    3) limited: The series that won Outstanding Limited or Anthology Series at the 77th Primetime Emmy Awards (2025).
       - title
       - platform
       - lead_actor (the person who won Outstanding Lead Actor in a Limited or Anthology Series for this show)
       - supporting_actor (the person who won Outstanding Supporting Actor in a Limited or Anthology Series for this show)
       - total_wins (confirm it is at least 8 total Emmy wins at the 2025 ceremony)
       - episode_count (number of episodes in the limited series)
       - urls (array of at least one URL referenced for this show)

    Rules:
    - Extract only what is explicitly present in the answer.
    - For all URLs fields, return every URL that the answer associates with that item; if none are provided, return an empty array.
    - Keep season_number, total_wins, and episode_count as strings exactly as written (e.g., "2", "10", "13", "Season 2"), do not coerce to integers.
    - Platforms should be extracted as they appear in the answer (e.g., "Apple TV+", "Netflix", "HBO Max", or "Max").

    Return a single JSON object with keys: drama, comedy, limited.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _ensure_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic cleanup and dedup
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        su = u.strip()
        if not su:
            continue
        if not (su.startswith("http://") or su.startswith("https://")):
            su = "http://" + su
        if su not in cleaned:
            cleaned.append(su)
    return cleaned


def canonicalize_platform(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    # Normalize common variants
    replacements = {
        "appletv+": "apple tv+",
        "apple tv plus": "apple tv+",
        "apple tv+ ": "apple tv+",
        "apple+ tv": "apple tv+",
        "max": "hbo max",   # treat "Max" as HBO Max
        "hbo": "hbo max",
        "hbo max": "hbo max",
        "net flix": "netflix",
        "netflix": "netflix",
    }
    # try exact mapping first
    if s in replacements:
        s = replacements[s]
    # minor cleanup for spacing
    s = " ".join(s.split())
    # final canonical forms
    if s in {"apple tv+"}:
        return "Apple TV+"
    if s in {"hbo max"}:
        return "HBO Max"
    if s in {"netflix"}:
        return "Netflix"
    return None


def platforms_distinct(p1: Optional[str], p2: Optional[str], p3: Optional[str] = None) -> Tuple[bool, List[str]]:
    normed = [canonicalize_platform(p) for p in [p1, p2, p3] if p is not None]
    # remove None
    normed = [p for p in normed if p is not None]
    issues = []
    if len(normed) != len({*normed}):
        issues.append("Platforms are not all distinct after normalization.")
        return False, issues
    return True, issues


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_drama(
    evaluator: Evaluator,
    parent_node,
    info: Optional[DramaSeriesInfo],
) -> Dict[str, Any]:
    # Safe defaults
    info = info or DramaSeriesInfo()
    urls = _ensure_urls(info.urls)
    norm_platform = canonicalize_platform(info.platform)

    # Category node
    drama_node = evaluator.add_parallel(
        id="drama_series",
        desc="Identify the series that won Outstanding Drama Series at the 2025 Primetime Emmy Awards",
        parent=parent_node,
        critical=False,
    )

    # 1) Title winner verification (critical)
    title_leaf = evaluator.add_leaf(
        id="drama_title",
        desc="The series title must match the Outstanding Drama Series winner",
        parent=drama_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The series that won Outstanding Drama Series at the 77th Primetime Emmy Awards (2025) is '{info.title}'.",
        node=title_leaf,
        sources=urls,
        additional_instruction="Use the provided sources to confirm the exact series that won the category. Accept references to '2025 Emmys' or '77th Primetime Emmy Awards' as equivalent.",
    )

    # 2) Platform allowed (critical) - membership only
    platform_allowed_node = evaluator.add_custom_node(
        result=(norm_platform in {"Apple TV+", "HBO Max", "Netflix"}),
        id="drama_platform",
        desc="The platform must be one of: Apple TV+, HBO Max, or Netflix",
        parent=drama_node,
        critical=True,
    )

    # 3) Season number presence (critical) then verification (critical)
    season_present_node = evaluator.add_custom_node(
        result=bool(info.season_number and str(info.season_number).strip()),
        id="drama_season_present",
        desc="Drama series season number is provided",
        parent=drama_node,
        critical=True,
    )
    season_verify_leaf = evaluator.add_leaf(
        id="drama_season_info",
        desc="Provide the season number that won the award",
        parent=drama_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Season '{info.season_number}' of '{info.title}' is the season that won Outstanding Drama Series at the 77th Primetime Emmy Awards (2025).",
        node=season_verify_leaf,
        sources=urls,
        additional_instruction="Confirm that the winning entry corresponds to the specified season number. Allow minor variants such as 'Season 2' vs '2'.",
        extra_prerequisites=[season_present_node],
    )

    # 4) Lead Actor in Drama (critical)
    lead_actor_leaf = evaluator.add_leaf(
        id="drama_lead_actor",
        desc="Identify the actor who won Outstanding Lead Actor in a Drama Series for this show",
        parent=drama_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{info.lead_actor}' won Outstanding Lead Actor in a Drama Series for '{info.title}' at the 77th Primetime Emmy Awards (2025).",
        node=lead_actor_leaf,
        sources=urls,
        additional_instruction="Verify the named performer is the Lead Actor winner and that the win is for this show.",
    )

    # 5) Supporting Actress in Drama (critical)
    supp_actress_leaf = evaluator.add_leaf(
        id="drama_supporting_actress",
        desc="Identify the actress who won Outstanding Supporting Actress in a Drama Series for this show",
        parent=drama_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{info.supporting_actress}' won Outstanding Supporting Actress in a Drama Series for '{info.title}' at the 77th Primetime Emmy Awards (2025).",
        node=supp_actress_leaf,
        sources=urls,
        additional_instruction="Verify the named performer is the Supporting Actress winner and that the win is for this show.",
    )

    # 6) Episode count (non-critical) - presence + verification
    ep_present = evaluator.add_custom_node(
        result=bool(info.episode_count and str(info.episode_count).strip()),
        id="drama_episode_count_present",
        desc="Drama winning season episode count is provided",
        parent=drama_node,
        critical=False,
    )
    ep_verify_leaf = evaluator.add_leaf(
        id="drama_episode_count",
        desc="Provide the number of episodes in the winning season",
        parent=drama_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The winning season ('{info.season_number}') of '{info.title}' has '{info.episode_count}' episodes.",
        node=ep_verify_leaf,
        sources=urls,
        additional_instruction="Confirm the number of episodes for the specific winning season. Allow reasonable formatting variants (e.g., numerals or words).",
        extra_prerequisites=[ep_present],
    )

    # 7) References: presence (critical) + win confirmation via source (critical)
    ref_present = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="drama_reference_presence",
        desc="At least one reference URL is provided for the Drama Series",
        parent=drama_node,
        critical=True,
    )
    ref_confirm = evaluator.add_leaf(
        id="drama_reference",
        desc="Provide at least one URL confirming the Drama Series win",
        parent=drama_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided sources confirm that '{info.title}' won Outstanding Drama Series at the 77th Primetime Emmy Awards (2025).",
        node=ref_confirm,
        sources=urls,
        additional_instruction="Decide only based on the provided URLs whether the show won the category.",
        extra_prerequisites=[ref_present],
    )

    return {
        "node": drama_node,
        "platform_norm": norm_platform,
        "platform_allowed_leaf": platform_allowed_node,
    }


async def verify_comedy(
    evaluator: Evaluator,
    parent_node,
    info: Optional[ComedySeriesInfo],
    drama_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    info = info or ComedySeriesInfo()
    urls = _ensure_urls(info.urls)
    norm_platform = canonicalize_platform(info.platform)
    drama_platform_norm = drama_ctx.get("platform_norm")

    comedy_node = evaluator.add_parallel(
        id="comedy_series",
        desc="Identify the series that won Outstanding Comedy Series at the 2025 Primetime Emmy Awards",
        parent=parent_node,
        critical=False,
    )

    # 1) Title winner verification (critical)
    title_leaf = evaluator.add_leaf(
        id="comedy_title",
        desc="The series title must match the Outstanding Comedy Series winner",
        parent=comedy_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The series that won Outstanding Comedy Series at the 77th Primetime Emmy Awards (2025) is '{info.title}'.",
        node=title_leaf,
        sources=urls,
        additional_instruction="Use the provided sources to confirm the exact series that won the category.",
    )

    # 2) Platform allowed (critical)
    platform_allowed_node = evaluator.add_custom_node(
        result=(norm_platform in {"Apple TV+", "HBO Max", "Netflix"}),
        id="comedy_platform_allowed",
        desc="The platform must be one of: Apple TV+, HBO Max, or Netflix",
        parent=comedy_node,
        critical=True,
    )

    # 3) Platform different from drama (critical)
    platform_distinct_leaf = evaluator.add_leaf(
        id="comedy_platform",
        desc="The platform must be different from the drama series and be one of: Apple TV+, HBO Max, or Netflix",
        parent=comedy_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The comedy series platform ('{norm_platform}') is different from the drama series platform ('{drama_platform_norm}').",
        node=platform_distinct_leaf,
        additional_instruction="Treat 'Max' as 'HBO Max'. Consider platforms equal if they normalize to the same one among {Apple TV+, HBO Max, Netflix}.",
        extra_prerequisites=[platform_allowed_node, drama_ctx.get("platform_allowed_leaf")],
    )

    # 4) Season number presence (critical) then verification (critical)
    season_present_node = evaluator.add_custom_node(
        result=bool(info.season_number and str(info.season_number).strip()),
        id="comedy_season_present",
        desc="Comedy series season number is provided",
        parent=comedy_node,
        critical=True,
    )
    season_verify_leaf = evaluator.add_leaf(
        id="comedy_season_info",
        desc="Provide the season number that won the award",
        parent=comedy_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Season '{info.season_number}' of '{info.title}' is the season that won Outstanding Comedy Series at the 77th Primetime Emmy Awards (2025).",
        node=season_verify_leaf,
        sources=urls,
        additional_instruction="Confirm that the winning entry corresponds to the specified season number. Allow minor variants such as 'Season 1' vs '1'.",
        extra_prerequisites=[season_present_node],
    )

    # 5) Lead Actor in Comedy (critical)
    lead_actor_leaf = evaluator.add_leaf(
        id="comedy_lead_actor",
        desc="Identify the actor who won Outstanding Lead Actor in a Comedy Series for this show",
        parent=comedy_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{info.lead_actor}' won Outstanding Lead Actor in a Comedy Series for '{info.title}' at the 77th Primetime Emmy Awards (2025).",
        node=lead_actor_leaf,
        sources=urls,
        additional_instruction="Verify the named performer is the Lead Actor (Comedy) winner and that the win is for this show.",
    )

    # 6) Emmy record (critical)
    record_leaf = evaluator.add_leaf(
        id="comedy_emmy_record",
        desc="Specify the Emmy record set by this series (must mention it was the most-winning freshman comedy in Emmy history)",
        parent=comedy_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The series '{info.title}' set the record as the most-winning freshman comedy in Emmy history at the 2025 ceremony.",
        node=record_leaf,
        sources=urls,
        additional_instruction="Look for explicit phrasing like 'most-winning freshman comedy in Emmy history' or an unmistakable paraphrase.",
    )

    # 7) Total wins == 13 (critical)
    total_wins_leaf = evaluator.add_leaf(
        id="comedy_total_wins",
        desc="Provide the total number of Emmy wins achieved by this series at the 2025 ceremony (must be 13)",
        parent=comedy_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{info.title}' won 13 total Emmy awards at the 77th Primetime Emmy Awards (2025), including Creative Arts categories.",
        node=total_wins_leaf,
        sources=urls,
        additional_instruction="Confirm the exact total is 13. Many articles aggregate Primetime + Creative Arts for the total.",
    )

    # 8) Episode count (non-critical) - presence + verification
    ep_present = evaluator.add_custom_node(
        result=bool(info.episode_count and str(info.episode_count).strip()),
        id="comedy_episode_count_present",
        desc="Comedy winning season episode count is provided",
        parent=comedy_node,
        critical=False,
    )
    ep_verify_leaf = evaluator.add_leaf(
        id="comedy_episode_count",
        desc="Provide the number of episodes in the winning season",
        parent=comedy_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The winning season ('{info.season_number}') of '{info.title}' has '{info.episode_count}' episodes.",
        node=ep_verify_leaf,
        sources=urls,
        additional_instruction="Confirm the number of episodes for the specified winning season.",
        extra_prerequisites=[ep_present],
    )

    # 9) References: presence (critical) + win confirmation via source (critical)
    ref_present = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="comedy_reference_presence",
        desc="At least one reference URL is provided for the Comedy Series",
        parent=comedy_node,
        critical=True,
    )
    ref_confirm = evaluator.add_leaf(
        id="comedy_reference",
        desc="Provide at least one URL confirming the Comedy Series win",
        parent=comedy_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided sources confirm that '{info.title}' won Outstanding Comedy Series at the 77th Primetime Emmy Awards (2025).",
        node=ref_confirm,
        sources=urls,
        additional_instruction="Decide only based on the provided URLs whether the show won the category.",
        extra_prerequisites=[ref_present],
    )

    return {
        "node": comedy_node,
        "platform_norm": norm_platform,
        "platform_allowed_leaf": platform_allowed_node,
    }


async def verify_limited(
    evaluator: Evaluator,
    parent_node,
    info: Optional[LimitedSeriesInfo],
    drama_ctx: Dict[str, Any],
    comedy_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    info = info or LimitedSeriesInfo()
    urls = _ensure_urls(info.urls)
    norm_platform = canonicalize_platform(info.platform)
    drama_platform_norm = drama_ctx.get("platform_norm")
    comedy_platform_norm = comedy_ctx.get("platform_norm")

    limited_node = evaluator.add_parallel(
        id="limited_series",
        desc="Identify the series that won Outstanding Limited or Anthology Series at the 2025 Primetime Emmy Awards",
        parent=parent_node,
        critical=False,
    )

    # 1) Title winner verification (critical)
    title_leaf = evaluator.add_leaf(
        id="limited_title",
        desc="The series title must match the Outstanding Limited or Anthology Series winner",
        parent=limited_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The series that won Outstanding Limited or Anthology Series at the 77th Primetime Emmy Awards (2025) is '{info.title}'.",
        node=title_leaf,
        sources=urls,
        additional_instruction="Use the provided sources to confirm the exact series that won the category.",
    )

    # 2) Platform allowed (critical)
    platform_allowed_node = evaluator.add_custom_node(
        result=(norm_platform in {"Apple TV+", "HBO Max", "Netflix"}),
        id="limited_platform_allowed",
        desc="The platform must be one of: Apple TV+, HBO Max, or Netflix",
        parent=limited_node,
        critical=True,
    )

    # 3) Platform different from both drama and comedy (critical)
    platform_distinct_leaf = evaluator.add_leaf(
        id="limited_platform",
        desc="The platform must be different from both the drama and comedy series and be one of: Apple TV+, HBO Max, or Netflix",
        parent=limited_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The limited series platform ('{norm_platform}') is different from the drama ('{drama_platform_norm}') and comedy ('{comedy_platform_norm}') series platforms.",
        node=platform_distinct_leaf,
        additional_instruction="Treat 'Max' as 'HBO Max'. Consider platforms equal if they normalize to the same one among {Apple TV+, HBO Max, Netflix}.",
        extra_prerequisites=[platform_allowed_node, drama_ctx.get("platform_allowed_leaf"), comedy_ctx.get("platform_allowed_leaf")],
    )

    # 4) Lead Actor in Limited/Anthology (critical)
    lead_actor_leaf = evaluator.add_leaf(
        id="limited_lead_actor",
        desc="Identify the actor who won Outstanding Lead Actor in a Limited or Anthology Series for this show",
        parent=limited_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{info.lead_actor}' won Outstanding Lead Actor in a Limited or Anthology Series for '{info.title}' at the 77th Primetime Emmy Awards (2025).",
        node=lead_actor_leaf,
        sources=urls,
        additional_instruction="Verify the named performer is the Lead Actor (Limited/Anthology) winner and that the win is for this show.",
    )

    # 5) Supporting Actor winner identity (critical)
    supporting_actor_identity_leaf = evaluator.add_leaf(
        id="limited_supporting_actor_identity",
        desc="Identify the actor who won Outstanding Supporting Actor in a Limited or Anthology Series for this show",
        parent=limited_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{info.supporting_actor}' won Outstanding Supporting Actor in a Limited or Anthology Series for '{info.title}' at the 77th Primetime Emmy Awards (2025).",
        node=supporting_actor_identity_leaf,
        sources=urls,
        additional_instruction="Verify the named performer is the Supporting Actor (Limited/Anthology) winner and that the win is for this show.",
    )

    # 6) Supporting Actor 'youngest' record (critical)
    supporting_actor_record_leaf = evaluator.add_leaf(
        id="limited_supporting_actor_record",
        desc="Confirm the Supporting Actor winner was 15 years old and became the youngest male Emmy winner in history",
        parent=limited_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Supporting Actor winner '{info.supporting_actor}' was 15 years old at the time and became the youngest male Emmy winner in history at the 77th Primetime Emmy Awards (2025).",
        node=supporting_actor_record_leaf,
        sources=urls,
        additional_instruction="Look for explicit statements of age (15) and the 'youngest male Emmy winner' record.",
    )

    # 7) Total wins at least 8 (critical)
    total_wins_leaf = evaluator.add_leaf(
        id="limited_total_wins",
        desc="Confirm the series won at least 8 Emmy awards at the 2025 ceremony",
        parent=limited_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{info.title}' won at least 8 total Emmy awards at the 77th Primetime Emmy Awards (2025), including Creative Arts.",
        node=total_wins_leaf,
        sources=urls,
        additional_instruction="If the exact number is provided, ensure it is >= 8. Consider both Primetime and Creative Arts for the total.",
    )

    # 8) Episode count (non-critical) - presence + verification
    ep_present = evaluator.add_custom_node(
        result=bool(info.episode_count and str(info.episode_count).strip()),
        id="limited_episode_count_present",
        desc="Limited series episode count is provided",
        parent=limited_node,
        critical=False,
    )
    ep_verify_leaf = evaluator.add_leaf(
        id="limited_episode_count",
        desc="Provide the number of episodes in the limited series",
        parent=limited_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The limited series '{info.title}' has '{info.episode_count}' total episodes.",
        node=ep_verify_leaf,
        sources=urls,
        additional_instruction="Confirm the total number of episodes in the limited/anthology series.",
        extra_prerequisites=[ep_present],
    )

    # 9) References: presence (critical) + win confirmation via source (critical)
    ref_present = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="limited_reference_presence",
        desc="At least one reference URL is provided for the Limited/Anthology Series",
        parent=limited_node,
        critical=True,
    )
    ref_confirm = evaluator.add_leaf(
        id="limited_reference",
        desc="Provide at least one URL confirming the Limited/Anthology Series win",
        parent=limited_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided sources confirm that '{info.title}' won Outstanding Limited or Anthology Series at the 77th Primetime Emmy Awards (2025).",
        node=ref_confirm,
        sources=urls,
        additional_instruction="Decide only based on the provided URLs whether the show won the category.",
        extra_prerequisites=[ref_present],
    )

    return {
        "node": limited_node,
        "platform_norm": norm_platform,
        "platform_allowed_leaf": platform_allowed_node,
    }


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
    """
    Evaluate an answer for the '77th Primetime Emmy Awards streaming winners' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: parallel aggregation of the three categories
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

    # 1) Extract structured winners info from the answer
    winners = await evaluator.extract(
        prompt=prompt_extract_winners(),
        template_class=WinnersExtraction,
        extraction_name="winners_extraction",
    )

    # 2) Add a quick normalized platform summary to the report
    evaluator.add_custom_info(
        info={
            "drama_platform_raw": (winners.drama.platform if winners.drama else None),
            "comedy_platform_raw": (winners.comedy.platform if winners.comedy else None),
            "limited_platform_raw": (winners.limited.platform if winners.limited else None),
            "drama_platform_norm": canonicalize_platform(winners.drama.platform if winners.drama else None),
            "comedy_platform_norm": canonicalize_platform(winners.comedy.platform if winners.comedy else None),
            "limited_platform_norm": canonicalize_platform(winners.limited.platform if winners.limited else None),
        },
        info_type="normalized_platforms",
    )

    # 3) Build verification subtrees for each category
    drama_ctx = await verify_drama(evaluator, root, winners.drama)
    comedy_ctx = await verify_comedy(evaluator, root, winners.comedy, drama_ctx)
    await verify_limited(evaluator, root, winners.limited, drama_ctx, comedy_ctx)

    # 4) Additional global distinctness check across all three (non-critical summary)
    distinct_ok, distinct_issues = platforms_distinct(
        winners.drama.platform if winners.drama else None,
        winners.comedy.platform if winners.comedy else None,
        winners.limited.platform if winners.limited else None,
    )
    evaluator.add_custom_node(
        result=distinct_ok,
        id="all_platforms_distinct",
        desc="All three winners are from different major streaming platforms",
        parent=root,
        critical=False,
    )
    if not distinct_ok and distinct_issues:
        evaluator.add_custom_info({"issues": distinct_issues}, info_type="platform_distinctness_issues")

    # 5) Return evaluation summary
    return evaluator.get_summary()