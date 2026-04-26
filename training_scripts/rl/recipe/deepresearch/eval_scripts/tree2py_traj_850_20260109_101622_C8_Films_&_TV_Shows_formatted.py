import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "emmys_2024_drama_platforms"
TASK_DESCRIPTION = (
    "Five streaming platforms—Apple TV+, Prime Video, Netflix, HBO, and FX—each had at least one drama series nominated for Outstanding Drama Series at the 76th Primetime Emmy Awards (2024). "
    "For each of these five platforms, identify one drama series that received this nomination and provide the following information: "
    "(1) The series title; (2) Confirmation of its nomination for Outstanding Drama Series at the 2024 Emmy Awards; "
    "(3) The streaming platform it aired on; (4) The number of episodes in the nominated season; "
    "(5) The name(s) of the series creator(s) or showrunner(s); (6) The premiere date or release period of the nominated season; "
    "(7) Whether the series won any acting Emmy Awards (Lead Actor, Lead Actress, Supporting Actor, or Supporting Actress in a Drama Series) at the 2024 ceremony, and if so, which actor(s) and for which role(s); "
    "(8) A reference URL that verifies the nomination."
)

ALLOWED_OPTIONS: Dict[str, List[str]] = {
    "apple_tv_plus": ["The Morning Show Season 3", "Slow Horses Season 3"],
    "prime_video": ["Fallout", "Mr. & Mrs. Smith"],
    "netflix": ["The Crown Season 6", "3 Body Problem"],
    "hbo": ["The Gilded Age Season 2"],
    "fx": ["Shōgun", "Shogun"]
}


class SeriesEntry(BaseModel):
    title: Optional[str] = None
    season: Optional[str] = None
    platform: Optional[str] = None
    nomination_reference_url: Optional[str] = None
    episode_count: Optional[str] = None
    creators_or_showrunners: List[str] = Field(default_factory=list)
    premiere_date_or_period: Optional[str] = None
    acting_emmys_summary: Optional[str] = None
    acting_emmys_details: List[str] = Field(default_factory=list)
    acting_emmys_reference_urls: List[str] = Field(default_factory=list)
    detail_source_urls: List[str] = Field(default_factory=list)


class EmmyNomineesExtraction(BaseModel):
    apple_tv_plus: Optional[SeriesEntry] = None
    prime_video: Optional[SeriesEntry] = None
    netflix: Optional[SeriesEntry] = None
    hbo: Optional[SeriesEntry] = None
    fx: Optional[SeriesEntry] = None


def prompt_extract_nominees() -> str:
    return (
        "You must extract one qualifying nominee entry for each of the five specified platforms (Apple TV+, Prime Video, Netflix, HBO, FX) as presented in the answer text. "
        "Only select from the allowed nominees per platform:\n"
        "- Apple TV+: The Morning Show Season 3 OR Slow Horses Season 3\n"
        "- Prime Video: Fallout OR Mr. & Mrs. Smith\n"
        "- Netflix: The Crown Season 6 OR 3 Body Problem\n"
        "- HBO: The Gilded Age Season 2\n"
        "- FX: Shōgun\n\n"
        "For each platform, extract a SeriesEntry object with the following fields strictly based on what the answer states:\n"
        "• title: Series title\n"
        "• season: The nominated season (e.g., 'Season 3') if applicable; otherwise null\n"
        "• platform: The streaming platform/network the series aired on (e.g., 'Apple TV+', 'Prime Video', 'Netflix', 'HBO', 'FX')\n"
        "• nomination_reference_url: A URL that explicitly verifies the series' nomination for Outstanding Drama Series at the 2024 Emmys (prefer an official Emmys/Television Academy page or credible trade publication)\n"
        "• episode_count: The number of episodes in the nominated season (keep as a string exactly as stated)\n"
        "• creators_or_showrunners: List of creator(s) or showrunner(s) names (each name as a separate string)\n"
        "• premiere_date_or_period: The premiere date or release period for the nominated season (string)\n"
        "• acting_emmys_summary: A concise statement indicating whether the series won any acting Emmys at the 2024 ceremony in drama categories, and if yes, what categories/actors/roles; if none, explicitly say 'None' or 'No acting wins'\n"
        "• acting_emmys_details: If applicable, list strings like 'Lead Actor: [Actor Name] as [Role]' etc.; otherwise return an empty list\n"
        "• acting_emmys_reference_urls: URLs that verify the acting Emmy awards information (winners or lack thereof), preferably Television Academy winners pages or credible sources\n"
        "• detail_source_urls: Additional URLs used to verify platform, episode count, creators/showrunners, and premiere date/period. These can include official platform pages, show pages, press releases, or reliable databases.\n\n"
        "Return a JSON object with keys: 'apple_tv_plus', 'prime_video', 'netflix', 'hbo', 'fx'. Each should be a SeriesEntry. "
        "If the answer does not provide the necessary information for a platform, set that platform's SeriesEntry fields to null or empty accordingly."
    )


def is_valid_url(u: Optional[str]) -> bool:
    if not u:
        return False
    s = u.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def source_list(entry: SeriesEntry) -> List[str]:
    urls: List[str] = []
    if entry.nomination_reference_url and is_valid_url(entry.nomination_reference_url):
        urls.append(entry.nomination_reference_url)
    for x in entry.detail_source_urls:
        if is_valid_url(x):
            urls.append(x)
    return urls


def acting_sources(entry: SeriesEntry) -> List[str]:
    urls: List[str] = []
    for x in entry.acting_emmys_reference_urls:
        if is_valid_url(x):
            urls.append(x)
    return urls


async def verify_platform_entry(
    evaluator: Evaluator,
    parent_node,
    entry: Optional[SeriesEntry],
    platform_key: str,
    platform_display: str,
    platform_node_desc: str
) -> None:
    platform_node = evaluator.add_parallel(
        id=f"{platform_key}_series",
        desc=platform_node_desc,
        parent=parent_node,
        critical=False
    )

    entry = entry or SeriesEntry()
    title = entry.title or ""
    season = entry.season or ""
    nomination_url = entry.nomination_reference_url or ""
    episode_count = entry.episode_count or ""
    creators = entry.creators_or_showrunners or []
    premiere = entry.premiere_date_or_period or ""
    acting_summary = (entry.acting_emmys_summary or "").strip()
    acting_details = entry.acting_emmys_details or []
    all_detail_sources = source_list(entry)
    acting_detail_sources = acting_sources(entry)

    allowed = ALLOWED_OPTIONS.get(platform_key, [])
    allowed_str = " OR ".join(allowed)

    # Eligible title choice (critical)
    eligible_leaf = evaluator.add_leaf(
        id=f"{platform_key}_eligible_title_choice",
        desc=f"Selected {platform_display} series is one of the allowed nominees ({allowed_str}).",
        parent=platform_node,
        critical=True
    )
    claim_eligible = (
        f"The chosen {platform_display} nominee matches one of the allowed options: {allowed_str}. "
        f"Chosen: title='{title}'"
        + (f", season='{season}'" if season else "")
        + ". Minor punctuation, diacritics, or season formatting differences should be considered equivalent."
    )
    await evaluator.verify(
        claim=claim_eligible,
        node=eligible_leaf,
        additional_instruction=(
            "Decide if the provided title (and season when specified) corresponds to one of the allowed options. "
            "Treat 'Shōgun' and 'Shogun' as equivalent; allow 'Season 3' vs 'S3' equivalence; ignore minor punctuation/case."
        )
    )

    # Series title provided (critical, existence)
    title_exists = evaluator.add_custom_node(
        result=bool(title.strip()),
        id=f"{platform_key}_series_title",
        desc="Series title is provided.",
        parent=platform_node,
        critical=True
    )

    # Reference URL for nomination provided (critical, existence)
    nomination_url_exists = evaluator.add_custom_node(
        result=is_valid_url(nomination_url),
        id=f"{platform_key}_reference_url_nomination",
        desc="Provides a reference URL that verifies the nomination for Outstanding Drama Series (2024).",
        parent=platform_node,
        critical=True
    )

    # Emmy nomination confirmation (critical, verify by nomination URL)
    nomination_leaf = evaluator.add_leaf(
        id=f"{platform_key}_emmy_nomination",
        desc="Confirms the series was nominated for Outstanding Drama Series at the 76th Primetime Emmy Awards (2024).",
        parent=platform_node,
        critical=True
    )
    claim_nomination = (
        f"The series '{title}'"
        + (f" ({season})" if season else "")
        + " was nominated for Outstanding Drama Series at the 2024 Primetime Emmy Awards."
    )
    await evaluator.verify(
        claim=claim_nomination,
        node=nomination_leaf,
        sources=nomination_url if is_valid_url(nomination_url) else None,
        additional_instruction=(
            "Use the provided nomination reference page to confirm that the series appears in the 2024 Outstanding Drama Series nominees list."
        )
    )

    # Platform verification (critical)
    platform_leaf = evaluator.add_leaf(
        id=f"{platform_key}_platform_verification",
        desc=f"Confirms the series aired on {platform_display}.",
        parent=platform_node,
        critical=True
    )
    claim_platform = (
        f"The series '{title}' aired on {platform_display}. Accept variants like 'Apple TV Plus' for 'Apple TV+', "
        f"'Amazon Prime Video' for 'Prime Video', and 'FX on Hulu' as FX network distribution."
    )
    await evaluator.verify(
        claim=claim_platform,
        node=platform_leaf,
        sources=all_detail_sources if all_detail_sources else (nomination_url if is_valid_url(nomination_url) else None),
        additional_instruction="Verify the streaming/network attribution aligns exactly with the requested platform."
    )

    # Episode count accuracy (critical)
    episodes_leaf = evaluator.add_leaf(
        id=f"{platform_key}_episode_count_accuracy",
        desc="Episode count for the nominated season is provided and matches the actual number of episodes in that nominated season.",
        parent=platform_node,
        critical=True
    )
    claim_episodes = (
        f"The nominated season"
        + (f" ({season})" if season else "")
        + f" of '{title}' has {episode_count} episodes."
    )
    await evaluator.verify(
        claim=claim_episodes,
        node=episodes_leaf,
        sources=all_detail_sources if all_detail_sources else None,
        additional_instruction="Check the season-specific episode count from reliable sources."
    )

    # Creators/showrunners accuracy (critical)
    creators_leaf = evaluator.add_leaf(
        id=f"{platform_key}_creators_accuracy",
        desc="Creator(s) or showrunner(s) are provided and accurate.",
        parent=platform_node,
        critical=True
    )
    creators_str = ", ".join(creators) if creators else ""
    claim_creators = (
        f"The listed creators/showrunner(s) for '{title}'"
        + (f" ({season})" if season else "")
        + f" are accurate: {creators_str}. The listed names must be actual creators or showrunners; the list need not be exhaustive."
    )
    await evaluator.verify(
        claim=claim_creators,
        node=creators_leaf,
        sources=all_detail_sources if all_detail_sources else None,
        additional_instruction=(
            "Confirm that all listed names are legitimate creators or showrunners of the series. "
            "It is acceptable if the list is not exhaustive, but incorrect names should cause failure."
        )
    )

    # Premiere date/period within eligibility window (critical)
    premiere_leaf = evaluator.add_leaf(
        id=f"{platform_key}_premiere_within_window",
        desc="Premiere date/release period of the nominated season is provided and falls within June 1, 2023–May 31, 2024.",
        parent=platform_node,
        critical=True
    )
    claim_premiere = (
        f"The premiere date or release period for the nominated season"
        + (f" ({season})" if season else "")
        + f" of '{title}' is '{premiere}', which falls within June 1, 2023–May 31, 2024."
    )
    await evaluator.verify(
        claim=claim_premiere,
        node=premiere_leaf,
        sources=all_detail_sources if all_detail_sources else None,
        additional_instruction=(
            "Verify the season's first release date or the stated release window and assess whether it lies within the eligibility window June 1, 2023–May 31, 2024."
        )
    )

    # Acting Emmys accuracy (critical)
    acting_leaf = evaluator.add_leaf(
        id=f"{platform_key}_acting_emmys_accuracy",
        desc="Accurately states whether the series won any 2024 acting Emmys (Lead/Supporting Actor/Actress in Drama); if yes, specifies category, actor name(s), and role(s).",
        parent=platform_node,
        critical=True
    )
    if acting_summary.lower() in {"none", "no", "no acting wins", "did not win"} or (not acting_summary and not acting_details):
        claim_acting = (
            f"At the 2024 Primetime Emmys, '{title}' did not win any acting awards in drama categories "
            f"(Lead Actor/Actress, Supporting Actor/Actress)."
        )
    else:
        details_str = "; ".join(acting_details) if acting_details else acting_summary
        claim_acting = (
            f"At the 2024 Primetime Emmys, '{title}' won the following acting award(s) in drama categories: {details_str}."
        )
    await evaluator.verify(
        claim=claim_acting,
        node=acting_leaf,
        sources=acting_detail_sources if acting_detail_sources else None,
        additional_instruction=(
            "Check Television Academy winners (or equally credible sources) for 2024 drama acting categories. "
            "If the series is not listed as a winner in any of those categories, the claim should be 'no acting wins'."
        )
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_nominees(),
        template_class=EmmyNomineesExtraction,
        extraction_name="emmy_nominees_2024_drama_by_platform"
    )

    platforms_meta = [
        ("apple_tv_plus", "Apple TV+", "Apple TV+ nominee entry"),
        ("prime_video", "Prime Video", "Prime Video nominee entry"),
        ("netflix", "Netflix", "Netflix nominee entry"),
        ("hbo", "HBO", "HBO nominee entry"),
        ("fx", "FX", "FX nominee entry"),
    ]

    for key, display, desc in platforms_meta:
        entry = getattr(extracted, key, None)
        await verify_platform_entry(
            evaluator=evaluator,
            parent_node=root,
            entry=entry,
            platform_key=key,
            platform_display=display,
            platform_node_desc=desc
        )

    return evaluator.get_summary()