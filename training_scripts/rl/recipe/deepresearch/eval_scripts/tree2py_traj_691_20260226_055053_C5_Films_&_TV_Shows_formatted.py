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
TASK_ID = "tv_drama_2025_2026_awards"
TASK_DESCRIPTION = (
    "I need to identify a U.S. television drama series that meets the following specific criteria: "
    "(1) The series must have premiered OR released a new season between January 1, 2025 and February 26, 2026; "
    "(2) The series must have won OR been nominated for at least one major television award (Emmy Award or Golden Globe Award) during either the 2024 or 2025 award ceremony season; "
    "(3) The series must be distributed on a major U.S. broadcast network (such as NBC, CBS, ABC, or Fox) OR a major streaming platform (such as Netflix, Peacock, Hulu, or similar); "
    "(4) The series must have an identifiable lead actor or lead actress in a starring role, and I need both the actor's name and the character name they portray; "
    "(5) The series must have publicly available viewership data, ratings information, or streaming performance metrics that demonstrate its audience reach; "
    "(6) The series must have a confirmed and verifiable episode count for the season that aired during the specified timeframe. "
    "Please provide the series name, network/platform, the specific season that aired in the timeframe with its premiere date and episode count, the award recognition details (award body, category, and win/nomination status), "
    "the lead actor/actress and their character name, and the viewership/performance data, along with reference URLs from reliable sources that verify each piece of information."
)

ALLOWED_MAJOR_NETWORKS = [
    "NBC", "CBS", "ABC", "FOX", "Fox"
]
ALLOWED_MAJOR_STREAMERS = [
    "Netflix", "Hulu", "Disney+", "Disney Plus", "Max", "HBO Max",
    "Amazon Prime Video", "Prime Video", "Peacock", "Paramount+", "Paramount Plus",
    "Apple TV+"
]

WINDOW_START = "January 1, 2025"
WINDOW_END = "February 26, 2026"
ALLOWED_AWARD_YEARS = ["2024", "2025"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SeriesTiming(BaseModel):
    series_name: Optional[str] = None
    distributor: Optional[str] = None  # Network/platform
    season: Optional[str] = None       # e.g., "Season 2"
    premiere_date: Optional[str] = None
    episode_count: Optional[str] = None
    timing_urls: List[str] = Field(default_factory=list)


class GenreInfo(BaseModel):
    primary_genre: Optional[str] = None
    genre_urls: List[str] = Field(default_factory=list)


class AwardInfo(BaseModel):
    award_body: Optional[str] = None       # "Emmy Awards" or "Golden Globe Awards"
    category: Optional[str] = None
    status: Optional[str] = None           # "Won" or "Nominated"
    year: Optional[str] = None             # "2024" or "2025"
    award_urls: List[str] = Field(default_factory=list)


class CastInfo(BaseModel):
    lead_actor: Optional[str] = None
    character_name: Optional[str] = None
    cast_urls: List[str] = Field(default_factory=list)


class ViewershipInfo(BaseModel):
    metric_description: Optional[str] = None  # e.g., "Episode 1 averaged 4.1 million viewers"
    metric_value: Optional[str] = None        # optional, free text or number
    viewership_urls: List[str] = Field(default_factory=list)


class SeriesExtraction(BaseModel):
    timing: Optional[SeriesTiming] = None
    genre: Optional[GenreInfo] = None
    award: Optional[AwardInfo] = None
    cast: Optional[CastInfo] = None
    viewership: Optional[ViewershipInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_all() -> str:
    return f"""
Extract the first TV drama series described in the answer that is intended to meet the specified criteria. 
Return a JSON with the following fields and nested sections. If any field is missing, set it to null (or an empty array for URL lists). 
Extract only information explicitly present in the answer, exactly as written. Extract the actual URLs for each category.

Required JSON schema:
{{
  "timing": {{
    "series_name": string|null,
    "distributor": string|null,             // The U.S. broadcast network or streaming platform (e.g., NBC, CBS, ABC, FOX, Netflix, Hulu, Disney+, Max, Prime Video, Peacock)
    "season": string|null,                  // e.g., "Season 2"
    "premiere_date": string|null,           // e.g., "January 15, 2025"
    "episode_count": string|null,           // e.g., "10"
    "timing_urls": string[]                 // URLs verifying name/network/season/premiere date/episode count
  }},
  "genre": {{
    "primary_genre": string|null,           // e.g., "crime drama", "procedural drama", "drama"
    "genre_urls": string[]                  // URLs that explicitly show the genre classification
  }},
  "award": {{
    "award_body": string|null,              // "Emmy Awards" or "Golden Globe Awards" (or equivalent phrasing)
    "category": string|null,                // e.g., "Best Television Series – Drama" or specific acting/writing category
    "status": string|null,                  // "Won" or "Nominated" (or equivalent wording)
    "year": string|null,                    // "2024" or "2025"
    "award_urls": string[]                  // URLs that show the nomination/win details
  }},
  "cast": {{
    "lead_actor": string|null,              // Lead actor/actress name
    "character_name": string|null,          // Character name portrayed by the lead
    "cast_urls": string[]                   // URLs showing the lead and character
  }},
  "viewership": {{
    "metric_description": string|null,      // e.g., "Episode 1 drew 3.2 million viewers (Nielsen)" or "Top 10 on Netflix for week of Feb 2"
    "metric_value": string|null,            // optional numeric or brief value extracted if present
    "viewership_urls": string[]             // URLs providing ratings/viewership/performance metrics
  }}
}}

Notes:
- If the answer presents multiple series, extract ONLY the first series.
- For each URL list, include only valid URLs explicitly present in the answer. 
- Do not invent any content or URLs.
- Keep dates as they appear in the answer (e.g., "January 5, 2025").
- The distributor should refer to the major U.S. network or major streaming platform stated in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_series_identification_and_timing(
    evaluator: Evaluator,
    parent,
    data: Optional[SeriesTiming],
) -> None:
    group = evaluator.add_parallel(
        id="Series_Identification_and_Timing",
        desc="Verify the series' basic information, release timing, and episode details",
        parent=parent,
        critical=True
    )

    series_name = data.series_name if data else None
    distributor = data.distributor if data else None
    season = data.season if data else None
    premiere_date = data.premiere_date if data else None
    episode_count = data.episode_count if data else None
    timing_urls = data.timing_urls if (data and data.timing_urls) else []

    # Reference URL existence (critical gate)
    ref_node = evaluator.add_custom_node(
        result=_urls_present(timing_urls),
        id="Series_Identification_and_Timing_Reference_URL",
        desc="Provide a URL from search results that verifies the series identification and timing information",
        parent=group,
        critical=True
    )

    # Series_Name
    series_name_node = evaluator.add_leaf(
        id="Series_Name",
        desc="Provide the official title of the TV drama series",
        parent=group,
        critical=True
    )
    if not _non_empty(series_name):
        series_name_node.score = 0.0
        series_name_node.status = "failed"
    else:
        claim = f"The official title of the series is '{series_name}'."
        await evaluator.verify(
            claim=claim,
            node=series_name_node,
            sources=timing_urls,
            additional_instruction="Verify that the page clearly names the series title. Allow reasonable punctuation or capitalization variations.",
            extra_prerequisites=[ref_node],
        )

    # Network_or_Platform
    network_node = evaluator.add_leaf(
        id="Network_or_Platform",
        desc="Identify the U.S. broadcast network or streaming platform that distributes the series",
        parent=group,
        critical=True
    )
    if not _non_empty(distributor):
        network_node.score = 0.0
        network_node.status = "failed"
    else:
        claim = (
            f"The series '{series_name or 'the series'}' is distributed in the U.S. on {distributor}, "
            f"which is a major broadcast network or major streaming platform."
        )
        allow_list = ", ".join(ALLOWED_MAJOR_NETWORKS + ALLOWED_MAJOR_STREAMERS)
        await evaluator.verify(
            claim=claim,
            node=network_node,
            sources=timing_urls,
            additional_instruction=(
                "Confirm that the page indicates the U.S. distribution network/platform. "
                f"Consider the platform valid if it is one of: {allow_list}. "
                "Treat synonyms as equivalent (e.g., 'Max' ≈ 'HBO Max'; 'Prime Video' ≈ 'Amazon Prime Video'; 'Disney+' ≈ 'Disney Plus')."
            ),
            extra_prerequisites=[ref_node],
        )

    # Season_and_Premiere_Date
    season_premiere_node = evaluator.add_leaf(
        id="Season_and_Premiere_Date",
        desc=f"Specify which season aired/premiered between {WINDOW_START} and {WINDOW_END}, and provide the exact premiere date",
        parent=group,
        critical=True
    )
    if not (_non_empty(season) and _non_empty(premiere_date)):
        season_premiere_node.score = 0.0
        season_premiere_node.status = "failed"
    else:
        claim = (
            f"Season '{season}' of '{series_name or 'the series'}' premiered on {premiere_date}, "
            f"which falls between {WINDOW_START} and {WINDOW_END} (inclusive)."
        )
        await evaluator.verify(
            claim=claim,
            node=season_premiere_node,
            sources=timing_urls,
            additional_instruction=(
                "Verify both the season designation and the premiere date. "
                f"Also check that the premiere date is within the inclusive window {WINDOW_START} to {WINDOW_END}."
            ),
            extra_prerequisites=[ref_node],
        )

    # Episode_Count
    episode_node = evaluator.add_leaf(
        id="Episode_Count",
        desc="State the total number of episodes in the specified season",
        parent=group,
        critical=True
    )
    if not _non_empty(episode_count):
        episode_node.score = 0.0
        episode_node.status = "failed"
    else:
        claim = (
            f"The specified season '{season or ''}' of '{series_name or 'the series'}' has {episode_count} episodes."
        ).strip()
        await evaluator.verify(
            claim=claim,
            node=episode_node,
            sources=timing_urls,
            additional_instruction="Verify that the page clearly indicates the total episode count for the specified season.",
            extra_prerequisites=[ref_node],
        )


async def verify_genre_classification(
    evaluator: Evaluator,
    parent,
    data: Optional[GenreInfo],
    series_name: Optional[str]
) -> None:
    group = evaluator.add_parallel(
        id="Genre_Classification",
        desc="Verify the series is primarily classified as drama, crime drama, or procedural drama",
        parent=parent,
        critical=True
    )

    primary_genre = data.primary_genre if data else None
    genre_urls = data.genre_urls if (data and data.genre_urls) else []

    # Reference URL existence (critical gate)
    ref_node = evaluator.add_custom_node(
        result=_urls_present(genre_urls),
        id="Genre_Classification_Reference_URL",
        desc="Provide a URL from search results that verifies the genre classification",
        parent=group,
        critical=True
    )

    # Primary_Genre
    genre_node = evaluator.add_leaf(
        id="Primary_Genre",
        desc="Confirm the series is primarily classified as drama, crime drama, or procedural drama genre",
        parent=group,
        critical=True
    )
    if not _non_empty(primary_genre):
        genre_node.score = 0.0
        genre_node.status = "failed"
    else:
        claim = (
            f"This source indicates that '{series_name or 'the series'}' is primarily a '{primary_genre}' series, "
            "which is within the drama family (e.g., drama, crime drama, procedural drama)."
        )
        await evaluator.verify(
            claim=claim,
            node=genre_node,
            sources=genre_urls,
            additional_instruction=(
                "Accept 'crime drama', 'police procedural', 'legal drama', 'medical drama', or similar as within drama. "
                "Minor wording variations are acceptable as long as the page clearly places it in a drama subgenre."
            ),
            extra_prerequisites=[ref_node],
        )


async def verify_award_recognition(
    evaluator: Evaluator,
    parent,
    data: Optional[AwardInfo],
    series_name: Optional[str]
) -> None:
    group = evaluator.add_parallel(
        id="Award_Recognition",
        desc="Verify the series received Emmy or Golden Globe recognition during the 2024-2025 award season",
        parent=parent,
        critical=True
    )

    award_body = data.award_body if data else None
    category = data.category if data else None
    status = data.status if data else None
    year = data.year if data else None
    award_urls = data.award_urls if (data and data.award_urls) else []

    # Reference URL existence (critical gate)
    ref_node = evaluator.add_custom_node(
        result=_urls_present(award_urls),
        id="Award_Recognition_Reference_URL",
        desc="Provide a URL from search results that verifies the award nomination or win",
        parent=group,
        critical=True
    )

    # Award_Body_and_Category
    body_cat_node = evaluator.add_leaf(
        id="Award_Body_and_Category",
        desc="Identify the specific award body and the category for which the series or its cast/crew was recognized",
        parent=group,
        critical=True
    )
    if not (_non_empty(award_body) and _non_empty(category)):
        body_cat_node.score = 0.0
        body_cat_node.status = "failed"
    else:
        claim = (
            f"This source shows recognition related to '{series_name or 'the series'}' from the {award_body} "
            f"in the category '{category}'."
        )
        await evaluator.verify(
            claim=claim,
            node=body_cat_node,
            sources=award_urls,
            additional_instruction=(
                "Recognition may pertain to the series itself or to its cast/crew. "
                "Ensure the award body is the Emmy Awards or Golden Globe Awards (or clear equivalent phrasing)."
            ),
            extra_prerequisites=[ref_node],
        )

    # Award_Status_and_Year
    status_year_node = evaluator.add_leaf(
        id="Award_Status_and_Year",
        desc="Specify whether it won or was nominated, and the year of the award ceremony (2024 or 2025)",
        parent=group,
        critical=True
    )
    if not (_non_empty(status) and _non_empty(year)):
        status_year_node.score = 0.0
        status_year_node.status = "failed"
    else:
        claim = (
            f"This source indicates that there was a {status.lower()} in {year} for the {award_body} related to "
            f"'{series_name or 'the series'}'. The year must be 2024 or 2025."
        )
        await evaluator.verify(
            claim=claim,
            node=status_year_node,
            sources=award_urls,
            additional_instruction=(
                "Confirm whether it was a win or nomination and verify the ceremony year is 2024 or 2025."
            ),
            extra_prerequisites=[ref_node],
        )


async def verify_lead_cast_information(
    evaluator: Evaluator,
    parent,
    data: Optional[CastInfo],
    series_name: Optional[str]
) -> None:
    group = evaluator.add_parallel(
        id="Lead_Cast_Information",
        desc="Identify the lead actor or actress starring in the series",
        parent=parent,
        critical=True
    )

    lead_actor = data.lead_actor if data else None
    character_name = data.character_name if data else None
    cast_urls = data.cast_urls if (data and data.cast_urls) else []

    # Reference URL existence (critical gate)
    ref_node = evaluator.add_custom_node(
        result=_urls_present(cast_urls),
        id="Lead_Cast_Information_Reference_URL",
        desc="Provide a URL from search results that verifies the lead cast information",
        parent=group,
        critical=True
    )

    # Lead_Actor_or_Actress_Name
    lead_actor_node = evaluator.add_leaf(
        id="Lead_Actor_or_Actress_Name",
        desc="Provide the name of the lead actor or actress in a starring role",
        parent=group,
        critical=True
    )
    if not _non_empty(lead_actor):
        lead_actor_node.score = 0.0
        lead_actor_node.status = "failed"
    else:
        claim = f"This page shows that {lead_actor} stars in '{series_name or 'the series'}' in a leading role."
        await evaluator.verify(
            claim=claim,
            node=lead_actor_node,
            sources=cast_urls,
            additional_instruction="Allow minor name variations (middle initials, accents). Confirm that the role is clearly a lead/starring role.",
            extra_prerequisites=[ref_node],
        )

    # Character_Name
    character_node = evaluator.add_leaf(
        id="Character_Name",
        desc="Provide the name of the character portrayed by the lead actor/actress",
        parent=group,
        critical=True
    )
    if not _non_empty(character_name) or not _non_empty(lead_actor):
        character_node.score = 0.0
        character_node.status = "failed"
    else:
        claim = f"This page shows that {lead_actor} portrays the character '{character_name}' in '{series_name or 'the series'}'."
        await evaluator.verify(
            claim=claim,
            node=character_node,
            sources=cast_urls,
            additional_instruction="Allow reasonable formatting differences for character names (with/without titles).",
            extra_prerequisites=[ref_node],
        )


async def verify_viewership_metrics(
    evaluator: Evaluator,
    parent,
    data: Optional[ViewershipInfo],
    series_name: Optional[str]
) -> None:
    group = evaluator.add_parallel(
        id="Viewership_or_Performance_Metrics",
        desc="Provide publicly available viewership, ratings, or performance data for the series",
        parent=parent,
        critical=True
    )

    metric_desc = data.metric_description if data else None
    view_urls = data.viewership_urls if (data and data.viewership_urls) else []

    # Reference URL existence (critical gate)
    ref_node = evaluator.add_custom_node(
        result=_urls_present(view_urls),
        id="Viewership_or_Performance_Metrics_Reference_URL",
        desc="Provide a URL from search results that verifies the viewership or performance data",
        parent=group,
        critical=True
    )

    # Viewership_Data
    view_node = evaluator.add_leaf(
        id="Viewership_Data",
        desc="State the viewership numbers, ratings performance, streaming metrics, or ranking information that demonstrates the series' audience reach",
        parent=group,
        critical=True
    )
    if not _non_empty(metric_desc):
        view_node.score = 0.0
        view_node.status = "failed"
    else:
        claim = f"This source provides audience performance data for '{series_name or 'the series'}': {metric_desc}"
        await evaluator.verify(
            claim=claim,
            node=view_node,
            sources=view_urls,
            additional_instruction=(
                "Accept Nielsen ratings, broadcast viewership reports, network/streamer press releases, or official "
                "streaming charts/rankings (e.g., Netflix Top 10). The statement must be clearly supported by the page."
            ),
            extra_prerequisites=[ref_node],
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level criteria are independent but all critical
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find one TV drama series that meets all specified criteria regarding timing, awards, cast, genre, and viewership during the 2025-2026 period",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_series_all(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction"
    )

    # Build top-level critical groups (must all pass)
    # All children under these critical groups must also be critical.
    await verify_series_identification_and_timing(
        evaluator=evaluator,
        parent=root,
        data=extracted.timing if extracted else None
    )

    await verify_genre_classification(
        evaluator=evaluator,
        parent=root,
        data=extracted.genre if extracted else None,
        series_name=extracted.timing.series_name if (extracted and extracted.timing) else None
    )

    await verify_award_recognition(
        evaluator=evaluator,
        parent=root,
        data=extracted.award if extracted else None,
        series_name=extracted.timing.series_name if (extracted and extracted.timing) else None
    )

    await verify_lead_cast_information(
        evaluator=evaluator,
        parent=root,
        data=extracted.cast if extracted else None,
        series_name=extracted.timing.series_name if (extracted and extracted.timing) else None
    )

    await verify_viewership_metrics(
        evaluator=evaluator,
        parent=root,
        data=extracted.viewership if extracted else None,
        series_name=extracted.timing.series_name if (extracted and extracted.timing) else None
    )

    # Add a small custom info record to aid debugging
    evaluator.add_custom_info(
        info={
            "allowed_major_networks": ALLOWED_MAJOR_NETWORKS,
            "allowed_major_streamers": ALLOWED_MAJOR_STREAMERS,
            "time_window": {"start": WINDOW_START, "end": WINDOW_END},
            "allowed_award_years": ALLOWED_AWARD_YEARS
        },
        info_type="policy_context",
        info_name="evaluation_policy"
    )

    return evaluator.get_summary()