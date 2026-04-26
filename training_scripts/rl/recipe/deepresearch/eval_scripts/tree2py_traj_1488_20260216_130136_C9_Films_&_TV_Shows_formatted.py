import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "emmys_2025_outstanding_drama_series_4pick"
TASK_DESCRIPTION = (
    "Identify 4 distinct drama television series from the 2025 Emmy Awards Outstanding Drama Series nominee category. "
    "Each series must meet ALL of the specified requirements for its slot (Series #1..#4). "
    "For each series, provide: title, evidence of meeting award criteria (wins/nominations), "
    "streaming platform verification with current availability, relevant production details "
    "(premiere date, episode count, franchise affiliation, or series format as applicable), "
    "and reference URLs supporting each claim."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ActorWin(BaseModel):
    actor_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class StreamingInfo(BaseModel):
    platform_names: List[str] = Field(default_factory=list)
    platform_urls: List[str] = Field(default_factory=list)


class ProductionDetails(BaseModel):
    premiere_date: Optional[str] = None
    premiere_urls: List[str] = Field(default_factory=list)

    season2_episode_count: Optional[str] = None
    episode_count_urls: List[str] = Field(default_factory=list)

    franchise: Optional[bool] = None
    franchise_name: Optional[str] = None
    franchise_urls: List[str] = Field(default_factory=list)

    anthology_or_category_change: Optional[bool] = None
    format_description: Optional[str] = None
    format_urls: List[str] = Field(default_factory=list)


class SeriesItem(BaseModel):
    title: Optional[str] = None

    emmy_win_urls: List[str] = Field(default_factory=list)
    emmy_nomination_urls: List[str] = Field(default_factory=list)

    # For Series #2 "most nominations" fact
    total_nomination_count: Optional[str] = None
    nomination_count_urls: List[str] = Field(default_factory=list)

    actor_win: Optional[ActorWin] = None
    streaming: Optional[StreamingInfo] = None
    production: Optional[ProductionDetails] = None


class FourSeriesExtraction(BaseModel):
    series1: Optional[SeriesItem] = None
    series2: Optional[SeriesItem] = None
    series3: Optional[SeriesItem] = None
    series4: Optional[SeriesItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    return """
Extract structured information for exactly four drama series as they correspond to Series #1, Series #2, Series #3, and Series #4 in the user's answer. If the answer lists more than four, only keep the first four intended for these slots. If fewer than four are present, set the missing ones to null.

For each series, extract the following fields exactly as they appear in the answer (do NOT invent):
- title: The series title string.

Shared fields (may be empty if not provided):
- actor_win: { actor_name, urls[] } — the name of at least one actor from the series who won an acting Emmy in 2025 (Lead/Supporting/Guest), and the reference URLs supporting the win and association with this series.
- streaming: { platform_names[], platform_urls[] } — platform names claimed (e.g., "HBO Max", "Max", "Apple TV+", "Disney+", "HBO") and reference URLs confirming current availability on the claimed platform(s). Include official platform pages when possible.
- production: {
    premiere_date, premiere_urls[],
    season2_episode_count, episode_count_urls[],
    franchise, franchise_name, franchise_urls[],
    anthology_or_category_change, format_description, format_urls[]
  }
  - premiere_date: as presented in the answer (any format).
  - season2_episode_count: as presented in the answer (string).
  - franchise: true/false if claimed that the series is part of a pre-existing franchise; else null.
  - franchise_name: the franchise name if provided (e.g., "Star Wars").
  - anthology_or_category_change: true/false if claimed that the series is an anthology with multiple seasons OR it previously competed in Limited Series and later moved to Drama; else null.
  - format_description: free-form description supporting the anthology/category-change claim.
  - For each production detail (premiere, episodes, franchise, format/type), include supporting reference URLs in the respective *_urls arrays.

Award/nominations-specific fields:
- emmy_win_urls[]: URLs supporting that the series won the 2025 Emmy for Outstanding Drama Series (Series #1).
- emmy_nomination_urls[]: URLs supporting that the series was nominated for the 2025 Emmy for Outstanding Drama Series (Series #3 and #4; may also be present for others).
- total_nomination_count: for Series #2, the total number of 2025 Emmy nominations reported in the answer for the series.
- nomination_count_urls[]: URLs supporting the 2025 nomination count and that it led the Outstanding Drama nominees.

IMPORTANT RULES:
- Only include URLs explicitly present in the answer text. If none are provided for a field, keep the corresponding array empty.
- Normalize obvious platform name variants in platform_names but do not invent names. Examples: "HBO Max" or "Max"; "Apple TV+" (aka "Apple TV Plus"); "Disney+".
- If a field is not present, set it to null (for scalars) or [] (for lists).
- Do not infer or search for new URLs.

Return a JSON with fields: series1, series2, series3, series4, each being a SeriesItem structure as defined.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return urls or []


def _domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def _any_domain_in(urls: List[str], allowed_suffixes: List[str]) -> bool:
    for u in urls:
        d = _domain(u)
        for suf in allowed_suffixes:
            if d.endswith(suf):
                return True
    return False


def credible_awards_source_present(urls: List[str]) -> bool:
    # Television Academy and broadly credible outlets
    allowed = [
        "emmys.com", "televisionacademy.com",  # official
        "variety.com", "hollywoodreporter.com", "deadline.com",
        "nytimes.com", "latimes.com", "washingtonpost.com",
        "bbc.com", "theguardian.com", "reuters.com", "apnews.com",
        "bloomberg.com", "cnbc.com", "rollingstone.com", "indiewire.com",
        "ew.com", "people.com", "usatoday.com", "cnn.com", "forbes.com"
    ]
    return _any_domain_in(_safe_list(urls), allowed)


def credible_platform_max(urls: List[str]) -> bool:
    allowed = ["max.com", "hbo.com", "hbomax.com"]
    return _any_domain_in(_safe_list(urls), allowed)


def credible_platform_apple(urls: List[str]) -> bool:
    allowed = ["tv.apple.com", "apple.com"]
    return _any_domain_in(_safe_list(urls), allowed)


def credible_platform_disney(urls: List[str]) -> bool:
    allowed = ["disneyplus.com", "disney.com"]
    return _any_domain_in(_safe_list(urls), allowed)


def credible_platform_hbo(urls: List[str]) -> bool:
    allowed = ["hbo.com", "hbomax.com", "max.com"]
    return _any_domain_in(_safe_list(urls), allowed)


def normalize_title_for_distinct(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return "".join(ch.lower() for ch in s if ch.isalnum())


def titles_are_distinct(items: List[Optional[SeriesItem]]) -> bool:
    normalized = []
    for it in items:
        key = normalize_title_for_distinct(it.title if it else None)
        if not key:
            # Missing title counts as not distinct/invalid
            return False
        normalized.append(key)
    return len(set(normalized)) == len(normalized)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_series_1(evaluator: Evaluator, parent, s: Optional[SeriesItem]) -> None:
    node = evaluator.add_parallel(
        id="Series_1",
        desc="First series must have won Outstanding Drama Series Emmy in 2025, have an acting Emmy winner, stream on HBO Max, and premiered in 2025",
        parent=parent,
        critical=False
    )
    title = s.title if s and s.title else "Series #1"

    # 1) Emmy Win (sequential; reference first to gate claim)
    win_seq = evaluator.add_sequential(
        id="Series_1_Emmy_Win",
        desc="Verify the series won the Emmy Award for Outstanding Drama Series in 2025",
        parent=node,
        critical=True
    )
    # Reference presence & credibility
    win_ref_ok = credible_awards_source_present(_safe_list(s.emmy_win_urls if s else []))
    evaluator.add_custom_node(
        result=win_ref_ok,
        id="Emmy_Win_Reference",
        desc="Provide reference URL(s) from Television Academy or credible news sources confirming the Outstanding Drama Series win",
        parent=win_seq,
        critical=True
    )
    win_leaf = evaluator.add_leaf(
        id="Won_Outstanding_Drama_2025",
        desc="Series is listed as the winner of Outstanding Drama Series at the 2025 Emmy Awards",
        parent=win_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{title}' won the 2025 Emmy Award for Outstanding Drama Series.",
        node=win_leaf,
        sources=_safe_list(s.emmy_win_urls if s else []),
        additional_instruction="Verify strictly against the provided webpages that this series is the 2025 Outstanding Drama Series winner (not just nominee). Prefer the Television Academy winners page; credible news outlets are acceptable."
    )

    # 2) Acting Emmy (sequential; reference first)
    actor_seq = evaluator.add_sequential(
        id="Series_1_Acting_Emmy",
        desc="Verify at least one actor from this series won an Emmy Award in any acting category in 2025",
        parent=node,
        critical=True
    )
    actor_urls = _safe_list(s.actor_win.urls if (s and s.actor_win) else [])
    actor_ref_ok = credible_awards_source_present(actor_urls)
    evaluator.add_custom_node(
        result=actor_ref_ok,
        id="Acting_Emmy_Reference",
        desc="Provide reference URL(s) confirming the actor's Emmy win and their association with this series",
        parent=actor_seq,
        critical=True
    )
    actor_leaf = evaluator.add_leaf(
        id="Actor_Won_Emmy_2025",
        desc="At least one actor from the series won an Emmy in an acting category (Lead Actor/Actress, Supporting Actor/Actress, or Guest Actor/Actress) in 2025",
        parent=actor_seq,
        critical=True
    )
    actor_name = s.actor_win.actor_name if (s and s.actor_win and s.actor_win.actor_name) else "an actor from this series"
    await evaluator.verify(
        claim=f"{actor_name} from the series '{title}' won a Primetime Emmy acting award in 2025 and the win is associated with this series.",
        node=actor_leaf,
        sources=actor_urls,
        additional_instruction="Confirm that the actor won in a Lead/Supporting/Guest acting category in 2025 and that the win is tied to this series (role or series mentioned)."
    )

    # 3) Streaming on HBO Max (sequential; reference first)
    stream_seq = evaluator.add_sequential(
        id="Series_1_Streaming_Platform",
        desc="Verify the series currently streams on HBO Max",
        parent=node,
        critical=True
    )
    platform_urls = _safe_list(s.streaming.platform_urls if (s and s.streaming) else [])
    platform_ref_ok = credible_platform_max(platform_urls)
    evaluator.add_custom_node(
        result=platform_ref_ok and len(platform_urls) > 0,
        id="Platform_Reference",
        desc="Provide reference URL(s) confirming HBO Max streaming availability",
        parent=stream_seq,
        critical=True
    )
    stream_leaf = evaluator.add_leaf(
        id="Streams_on_HBO_Max",
        desc="Series is available to stream on the HBO Max platform",
        parent=stream_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{title}' is currently available to stream on HBO Max (also known as Max).",
        node=stream_leaf,
        sources=platform_urls,
        additional_instruction="Treat 'Max' as equivalent to 'HBO Max'. Verify that the official platform page or credible listing indicates current availability."
    )

    # 4) Premiered in 2025 (sequential; reference first)
    prem_seq = evaluator.add_sequential(
        id="Series_1_Premiere_Date",
        desc="Verify the series premiered in 2025",
        parent=node,
        critical=True
    )
    prem_urls = _safe_list(s.production.premiere_urls if (s and s.production) else [])
    evaluator.add_custom_node(
        result=len(prem_urls) > 0,
        id="Premiere_Reference",
        desc="Provide reference URL(s) confirming the premiere date in 2025",
        parent=prem_seq,
        critical=True
    )
    premiered_leaf = evaluator.add_leaf(
        id="Premiered_2025",
        desc="Series premiere date falls within the year 2025",
        parent=prem_seq,
        critical=True
    )
    prem_date_text = s.production.premiere_date if (s and s.production and s.production.premiere_date) else "in 2025"
    await evaluator.verify(
        claim=f"The series '{title}' premiered in 2025 (premiere date stated as {prem_date_text}).",
        node=premiered_leaf,
        sources=prem_urls,
        additional_instruction="Confirm the first-ever series premiere occurred in the calendar year 2025. If multiple dates exist (international vs domestic), accept if any official premiere is in 2025."
    )


async def build_series_2(evaluator: Evaluator, parent, s: Optional[SeriesItem]) -> None:
    node = evaluator.add_parallel(
        id="Series_2",
        desc="Second series must have received the most Emmy nominations, stream on Apple TV+, have Season 2 with 10 episodes, and have an acting Emmy winner",
        parent=parent,
        critical=False
    )
    title = s.title if s and s.title else "Series #2"

    # 1) Most nominations among Outstanding Drama Series nominees (sequential; reference first)
    noms_seq = evaluator.add_sequential(
        id="Series_2_Most_Nominations",
        desc="Verify the series received the highest number of Emmy nominations among all Outstanding Drama Series nominees in 2025",
        parent=node,
        critical=True
    )
    count_urls = _safe_list(s.nomination_count_urls if s else [])
    evaluator.add_custom_node(
        result=credible_awards_source_present(count_urls),
        id="Nomination_Count_Reference",
        desc="Provide reference URL(s) confirming the total nomination count and comparison to other nominees",
        parent=noms_seq,
        critical=True
    )
    highest_leaf = evaluator.add_leaf(
        id="Highest_Nomination_Count",
        desc="Series received more Emmy nominations than any other Outstanding Drama Series nominee in 2025",
        parent=noms_seq,
        critical=True
    )
    count_text = s.total_nomination_count if (s and s.total_nomination_count) else "the highest number of"
    await evaluator.verify(
        claim=f"The series '{title}' received {count_text} Emmy nominations among all 2025 Outstanding Drama Series nominees (i.e., it led the Drama Series nominees).",
        node=highest_leaf,
        sources=count_urls,
        additional_instruction="Focus on the 2025 Outstanding Drama Series nominees cohort and confirm this series led them in total nominations. Wording like 'led drama nominees' counts as confirmation."
    )

    # 2) Streams on Apple TV+ (sequential; reference first)
    apple_seq = evaluator.add_sequential(
        id="Series_2_Apple_TV_Plus",
        desc="Verify the series streams on Apple TV+",
        parent=node,
        critical=True
    )
    platform_urls = _safe_list(s.streaming.platform_urls if (s and s.streaming) else [])
    evaluator.add_custom_node(
        result=credible_platform_apple(platform_urls) and len(platform_urls) > 0,
        id="Apple_Platform_Reference",
        desc="Provide reference URL(s) confirming Apple TV+ streaming availability",
        parent=apple_seq,
        critical=True
    )
    apple_leaf = evaluator.add_leaf(
        id="Streams_on_Apple_TV_Plus",
        desc="Series is available to stream on the Apple TV+ platform",
        parent=apple_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{title}' is available to stream on Apple TV+.",
        node=apple_leaf,
        sources=platform_urls,
        additional_instruction="Treat 'Apple TV Plus' as equivalent to 'Apple TV+'. Prefer tv.apple.com pages indicating availability."
    )

    # 3) Season 2 has exactly 10 episodes (sequential; reference first)
    ep_seq = evaluator.add_sequential(
        id="Series_2_Episode_Count",
        desc="Verify the second season contains exactly 10 episodes",
        parent=node,
        critical=True
    )
    ep_urls = _safe_list(s.production.episode_count_urls if (s and s.production) else [])
    evaluator.add_custom_node(
        result=len(ep_urls) > 0,
        id="Episode_Count_Reference",
        desc="Provide reference URL(s) confirming the episode count for Season 2",
        parent=ep_seq,
        critical=True
    )
    ep_leaf = evaluator.add_leaf(
        id="Season_2_Has_10_Episodes",
        desc="Second season of the series consists of exactly 10 episodes",
        parent=ep_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"Season 2 of '{title}' has exactly 10 episodes.",
        node=ep_leaf,
        sources=ep_urls,
        additional_instruction="Verify explicitly for Season 2's episode count equals 10. Accept official or widely trusted sources (platform page, production notes, credible databases)."
    )

    # 4) Acting Emmy (sequential; reference first)
    actor_seq = evaluator.add_sequential(
        id="Series_2_Acting_Emmy",
        desc="Verify at least one actor from this series won an Emmy Award in any acting category in 2025",
        parent=node,
        critical=True
    )
    actor_urls = _safe_list(s.actor_win.urls if (s and s.actor_win) else [])
    evaluator.add_custom_node(
        result=credible_awards_source_present(actor_urls),
        id="Acting_Emmy_Reference",
        desc="Provide reference URL(s) confirming the actor's Emmy win and their association with this series",
        parent=actor_seq,
        critical=True
    )
    actor_leaf = evaluator.add_leaf(
        id="Actor_Won_Emmy_2025",
        desc="At least one actor from the series won an Emmy in an acting category in 2025",
        parent=actor_seq,
        critical=True
    )
    actor_name = s.actor_win.actor_name if (s and s.actor_win and s.actor_win.actor_name) else "an actor from this series"
    await evaluator.verify(
        claim=f"{actor_name} from the series '{title}' won a Primetime Emmy acting award in 2025 and the win is associated with this series.",
        node=actor_leaf,
        sources=actor_urls,
        additional_instruction="Confirm a Lead/Supporting/Guest acting win in 2025 tied to this series."
    )


async def build_series_3(evaluator: Evaluator, parent, s: Optional[SeriesItem]) -> None:
    node = evaluator.add_parallel(
        id="Series_3",
        desc="Third series must be nominated for Outstanding Drama Series, be part of an established franchise, and stream on Disney+",
        parent=parent,
        critical=False
    )
    title = s.title if s and s.title else "Series #3"

    # 1) Emmy Nomination (sequential; reference first)
    nom_seq = evaluator.add_sequential(
        id="Series_3_Emmy_Nomination",
        desc="Verify the series was nominated for Outstanding Drama Series in 2025",
        parent=node,
        critical=True
    )
    nom_urls = _safe_list(s.emmy_nomination_urls if s else [])
    evaluator.add_custom_node(
        result=credible_awards_source_present(nom_urls),
        id="Nomination_Reference",
        desc="Provide reference URL(s) confirming the Outstanding Drama Series nomination",
        parent=nom_seq,
        critical=True
    )
    nom_leaf = evaluator.add_leaf(
        id="Nominated_Outstanding_Drama",
        desc="Series is listed among the nominees for Outstanding Drama Series at the 2025 Emmy Awards",
        parent=nom_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{title}' was nominated for the 2025 Emmy Award for Outstanding Drama Series.",
        node=nom_leaf,
        sources=nom_urls,
        additional_instruction="Prefer the Television Academy nominees page; credible news outlets are acceptable."
    )

    # 2) Franchise status (sequential; reference first)
    fran_seq = evaluator.add_sequential(
        id="Series_3_Franchise_Status",
        desc="Verify the series is part of an established entertainment franchise",
        parent=node,
        critical=True
    )
    fran_urls = _safe_list(s.production.franchise_urls if (s and s.production) else [])
    evaluator.add_custom_node(
        result=len(fran_urls) > 0,
        id="Franchise_Reference",
        desc="Provide reference URL(s) confirming the franchise affiliation and that the franchise existed before the series",
        parent=fran_seq,
        critical=True
    )
    fran_leaf = evaluator.add_leaf(
        id="Part_of_Established_Franchise",
        desc="Series is part of a pre-existing film series, shared universe, or existing intellectual property (e.g., Star Wars, Marvel, etc.)",
        parent=fran_seq,
        critical=True
    )
    franchise_name = s.production.franchise_name if (s and s.production and s.production.franchise_name) else "an established franchise"
    await evaluator.verify(
        claim=f"The series '{title}' is part of {franchise_name}, an established franchise that predates the series.",
        node=fran_leaf,
        sources=fran_urls,
        additional_instruction="Confirm that the franchise (IP/universe) existed before the series debuted (films, books, prior TV, etc.)."
    )

    # 3) Streams on Disney+ (sequential; reference first)
    dplus_seq = evaluator.add_sequential(
        id="Series_3_Disney_Plus",
        desc="Verify the series streams on Disney+",
        parent=node,
        critical=True
    )
    platform_urls = _safe_list(s.streaming.platform_urls if (s and s.streaming) else [])
    evaluator.add_custom_node(
        result=credible_platform_disney(platform_urls) and len(platform_urls) > 0,
        id="Disney_Platform_Reference",
        desc="Provide reference URL(s) confirming Disney+ streaming availability",
        parent=dplus_seq,
        critical=True
    )
    dplus_leaf = evaluator.add_leaf(
        id="Streams_on_Disney_Plus",
        desc="Series is available to stream on the Disney+ platform",
        parent=dplus_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{title}' is available to stream on Disney+.",
        node=dplus_leaf,
        sources=platform_urls,
        additional_instruction="Prefer official disneyplus.com pages indicating availability."
    )


async def build_series_4(evaluator: Evaluator, parent, s: Optional[SeriesItem]) -> None:
    node = evaluator.add_parallel(
        id="Series_4",
        desc="Fourth series must be nominated for Outstanding Drama Series, be an anthology series or have moved from Limited Series category, and stream on HBO",
        parent=parent,
        critical=False
    )
    title = s.title if s and s.title else "Series #4"

    # 1) Emmy Nomination (sequential; reference first)
    nom_seq = evaluator.add_sequential(
        id="Series_4_Emmy_Nomination",
        desc="Verify the series was nominated for Outstanding Drama Series in 2025",
        parent=node,
        critical=True
    )
    nom_urls = _safe_list(s.emmy_nomination_urls if s else [])
    evaluator.add_custom_node(
        result=credible_awards_source_present(nom_urls),
        id="Nomination_Reference",
        desc="Provide reference URL(s) confirming the Outstanding Drama Series nomination",
        parent=nom_seq,
        critical=True
    )
    nom_leaf = evaluator.add_leaf(
        id="Nominated_Outstanding_Drama",
        desc="Series is listed among the nominees for Outstanding Drama Series at the 2025 Emmy Awards",
        parent=nom_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{title}' was nominated for the 2025 Emmy Award for Outstanding Drama Series.",
        node=nom_leaf,
        sources=nom_urls,
        additional_instruction="Prefer the Television Academy nominees page; credible news outlets are acceptable."
    )

    # 2) Anthology or category change (sequential; reference first)
    fmt_seq = evaluator.add_sequential(
        id="Series_4_Format_Type",
        desc="Verify the series is either an anthology series with multiple seasons OR previously competed in Limited Series category before moving to Drama Series",
        parent=node,
        critical=True
    )
    fmt_urls = _safe_list(s.production.format_urls if (s and s.production) else [])
    evaluator.add_custom_node(
        result=len(fmt_urls) > 0,
        id="Format_Reference",
        desc="Provide reference URL(s) confirming the anthology format or category change from Limited Series to Drama Series",
        parent=fmt_seq,
        critical=True
    )
    fmt_leaf = evaluator.add_leaf(
        id="Anthology_or_Category_Change",
        desc="Series meets at least one of: (1) is an anthology with multiple seasons, or (2) previously competed in Emmy Limited Series category and later moved to Drama Series category",
        parent=fmt_seq,
        critical=True
    )
    fmt_desc = s.production.format_description if (s and s.production and s.production.format_description) else "anthology or category change"
    await evaluator.verify(
        claim=f"The series '{title}' satisfies: {fmt_desc} — that is, it is an anthology with multiple seasons or it previously competed as a Limited Series and later moved to Drama Series.",
        node=fmt_leaf,
        sources=fmt_urls,
        additional_instruction="Either condition is sufficient. Confirm via the provided references."
    )

    # 3) Streams on HBO/HBO Max (sequential; reference first)
    hbo_seq = evaluator.add_sequential(
        id="Series_4_HBO_Platform",
        desc="Verify the series streams on HBO or HBO Max",
        parent=node,
        critical=True
    )
    platform_urls = _safe_list(s.streaming.platform_urls if (s and s.streaming) else [])
    evaluator.add_custom_node(
        result=credible_platform_hbo(platform_urls) and len(platform_urls) > 0,
        id="HBO_Platform_Reference",
        desc="Provide reference URL(s) confirming HBO or HBO Max streaming availability",
        parent=hbo_seq,
        critical=True
    )
    hbo_leaf = evaluator.add_leaf(
        id="Streams_on_HBO",
        desc="Series is available to stream on HBO or HBO Max",
        parent=hbo_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{title}' is available to stream on HBO or HBO Max (Max).",
        node=hbo_leaf,
        sources=platform_urls,
        additional_instruction="Treat 'Max' as acceptable evidence of availability under HBO/HBO Max branding."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=FourSeriesExtraction,
        extraction_name="four_series_extraction",
    )

    # Record a custom info summary of extracted titles
    evaluator.add_custom_info(
        info={
            "series1_title": extracted.series1.title if extracted.series1 else None,
            "series2_title": extracted.series2.title if extracted.series2 else None,
            "series3_title": extracted.series3.title if extracted.series3 else None,
            "series4_title": extracted.series4.title if extracted.series4 else None,
        },
        info_type="extraction_summary",
        info_name="extracted_series_titles"
    )

    # Add a critical distinctness check at root
    distinct_ok = titles_are_distinct([extracted.series1, extracted.series2, extracted.series3, extracted.series4])
    evaluator.add_custom_node(
        result=distinct_ok,
        id="All_Titles_Distinct",
        desc="All four series titles are present and distinct",
        parent=root,
        critical=True
    )

    # Build verification subtrees for each series
    await build_series_1(evaluator, root, extracted.series1)
    await build_series_2(evaluator, root, extracted.series2)
    await build_series_3(evaluator, root, extracted.series3)
    await build_series_4(evaluator, root, extracted.series4)

    return evaluator.get_summary()