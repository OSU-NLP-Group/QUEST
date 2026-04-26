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
TASK_ID = "early2025_scripted_series"
TASK_DESCRIPTION = """
Identify a scripted drama or narrative TV series that premiered between January 1, 2025, and March 31, 2025, on one of the major streaming platforms (Netflix, Hulu, Apple TV+, or Max). For this series, provide comprehensive information including: (1) the exact premiere date, (2) the streaming platform, (3) the total number of episodes in the season that premiered during this period (which must be between 8-15 episodes), (4) at least one genre classification, (5) which season premiered in early 2025, (6) at least one lead cast member, (7) the creator, showrunner, or executive producer, (8) the primary filming location if real-world location filming was used, (9) whether episodes were released weekly or all-at-once, (10) the current renewal status, (11) the approximate runtime per episode in minutes, and (12) reference URLs from the streaming platform or reliable entertainment news sources supporting each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SeriesInfo(BaseModel):
    # Core identity
    title: Optional[str] = None

    # Core fields + per-field sources
    premiere_date: Optional[str] = None
    premiere_sources: List[str] = Field(default_factory=list)

    platform: Optional[str] = None
    platform_sources: List[str] = Field(default_factory=list)

    episode_count: Optional[str] = None
    episode_count_sources: List[str] = Field(default_factory=list)

    genres: List[str] = Field(default_factory=list)
    genre_sources: List[str] = Field(default_factory=list)

    season_number: Optional[str] = None
    season_sources: List[str] = Field(default_factory=list)

    lead_cast: List[str] = Field(default_factory=list)
    cast_sources: List[str] = Field(default_factory=list)

    creator_or_showrunner_or_ep: Optional[str] = None
    creator_sources: List[str] = Field(default_factory=list)

    filming_location: Optional[str] = None
    filming_sources: List[str] = Field(default_factory=list)

    release_format: Optional[str] = None  # e.g., "weekly", "all-at-once"
    release_sources: List[str] = Field(default_factory=list)

    renewal_status: Optional[str] = None  # e.g., "renewed", "cancelled", "pending"
    renewal_sources: List[str] = Field(default_factory=list)

    series_type: Optional[str] = None  # e.g., "scripted drama", "crime drama", etc.
    series_type_sources: List[str] = Field(default_factory=list)

    episode_runtime: Optional[str] = None  # approximate minutes; string for robustness
    runtime_sources: List[str] = Field(default_factory=list)

    # Global references mentioned anywhere in the answer
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_info() -> str:
    return """
    Extract structured information about a single scripted drama or narrative TV series described in the answer.
    IMPORTANT: Extract ONLY what is explicitly present in the answer text. Do not invent or infer.

    Fields to extract:
    1) title: The series title.
    2) premiere_date: The exact date the season premiered in early 2025 (Jan 1 to Mar 31, 2025), as written in the answer.
       premiere_sources: URLs cited in the answer that directly support the premiere date.
    3) platform: The streaming platform (Netflix, Hulu, Apple TV+, or Max) as named in the answer.
       platform_sources: URLs cited that support the platform information (prefer platform or show page).
    4) episode_count: The total number of episodes in the season that premiered during this period (as written).
       episode_count_sources: URLs supporting the episode count for that season.
    5) genres: An array of at least one genre label mentioned (e.g., "drama", "crime", "mystery"; use the exact words from the answer).
       genre_sources: URLs supporting the genre classification.
    6) season_number: Which season premiered in early 2025 (e.g., "Season 1", "Season 2"), as written.
       season_sources: URLs supporting the season number.
    7) lead_cast: An array with at least one lead cast member named in the answer.
       cast_sources: URLs supporting the cast member(s).
    8) creator_or_showrunner_or_ep: The identified creator, showrunner, or executive producer (string from the answer).
       creator_sources: URLs supporting this credit.
    9) filming_location: The primary filming location if real-world location filming was used, as named in the answer.
       filming_sources: URLs supporting this location.
    10) release_format: Whether episodes were released "weekly" or "all-at-once" (use these exact normalized strings if clearly indicated).
        release_sources: URLs supporting the release model/schedule.
    11) renewal_status: The current renewal status mentioned in the answer (e.g., "renewed", "cancelled", "pending", or a short descriptive phrase).
        renewal_sources: URLs supporting this status.
    12) series_type: A short descriptor that makes clear it's a scripted drama or narrative TV series (not reality/doc/talk).
        series_type_sources: URLs supporting the type classification.
    13) episode_runtime: Approximate runtime per episode in minutes (as a short string, e.g., "45–50", "about 60").
        runtime_sources: URLs supporting the runtime.

    Additionally:
    - reference_urls: Extract ALL URLs mentioned anywhere in the answer (including platform and news sites).

    SPECIAL RULES:
    - For any per-field sources list, include all URLs explicitly cited that support that field.
    - If the answer provides no URL for a field, return an empty list for that field’s sources.
    - If a field value is missing in the answer, set it to null (or an empty array for list fields).
    - Do not guess or infer missing info.

    Return a JSON object matching the SeriesInfo schema exactly.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


def _first_or_empty(items: List[str]) -> str:
    return items[0] if items else ""


def _pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    return primary if (primary and len(primary) > 0) else (fallback if fallback else [])


def _join_list(values: List[str]) -> str:
    return ", ".join([v for v in values if _has_text(v)])


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_series_info(evaluator: Evaluator, root_node, info: SeriesInfo) -> None:
    # 1) Premiere date
    prem_node = evaluator.add_sequential(
        id="premiere_date",
        desc="The series premiered between January 1, 2025, and March 31, 2025",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.premiere_date),
        id="premiere_date_provided",
        desc="Premiere date is provided in the answer",
        parent=prem_node,
        critical=True,
    )
    prem_verify = evaluator.add_leaf(
        id="premiere_date_verified",
        desc="Premiere date is correct and within Jan 1–Mar 31, 2025",
        parent=prem_node,
        critical=True,
    )
    prem_claim = f"The series premiered on {info.premiere_date}. That date falls between January 1, 2025 and March 31, 2025 (inclusive)."
    await evaluator.verify(
        claim=prem_claim,
        node=prem_verify,
        sources=_pick_sources(info.premiere_sources, info.reference_urls),
        additional_instruction="Verify the premiere date and confirm it lies within the stated 2025 window. If multiple dates are shown (e.g., press events or regional releases), use the streaming premiere date on the stated platform."
    )

    # 2) Streaming platform
    plat_node = evaluator.add_sequential(
        id="streaming_platform",
        desc="The series streams on Netflix, Hulu, Apple TV+, or Max (HBO Max)",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.platform),
        id="streaming_platform_provided",
        desc="Streaming platform is provided",
        parent=plat_node,
        critical=True,
    )
    plat_verify = evaluator.add_leaf(
        id="streaming_platform_verified",
        desc="Streaming platform is one of Netflix, Hulu, Apple TV+, or Max and is correct",
        parent=plat_node,
        critical=True,
    )
    plat_claim = f"The streaming platform of the series is {info.platform}, and it is one of Netflix, Hulu, Apple TV+, or Max (HBO Max)."
    await evaluator.verify(
        claim=plat_claim,
        node=plat_verify,
        sources=_pick_sources(info.platform_sources, info.reference_urls),
        additional_instruction="Accept 'Max' or 'HBO Max' as the same service. Confirm the platform shown on an official show page or reliable coverage."
    )

    # 3) Episode count (8–15)
    epc_node = evaluator.add_sequential(
        id="episode_count",
        desc="The answer specifies a clear episode count between 8-15 episodes for the season that premiered in early 2025",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.episode_count),
        id="episode_count_provided",
        desc="Episode count is provided",
        parent=epc_node,
        critical=True,
    )
    epc_verify = evaluator.add_leaf(
        id="episode_count_verified",
        desc="Episode count is correct for the early-2025 season and between 8–15",
        parent=epc_node,
        critical=True,
    )
    epc_claim = f"The season that premiered in early 2025 has {info.episode_count} episodes, and that number lies between 8 and 15 inclusive."
    await evaluator.verify(
        claim=epc_claim,
        node=epc_verify,
        sources=_pick_sources(info.episode_count_sources, info.reference_urls),
        additional_instruction="Confirm the episode count specifically for the season that premiered between Jan–Mar 2025. Allow reasonable phrasing like 'ten-episode season' or similar."
    )

    # 4) Genre
    gen_node = evaluator.add_sequential(
        id="genre",
        desc="The answer identifies at least one genre classification for the series",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info.genres),
        id="genre_provided",
        desc="At least one genre is provided",
        parent=gen_node,
        critical=True,
    )
    gen_verify = evaluator.add_leaf(
        id="genre_verified",
        desc="The listed genre classification is correct",
        parent=gen_node,
        critical=True,
    )
    genres_joined = _join_list(info.genres)
    gen_claim = f"The series is classified with genre(s) including: {genres_joined}."
    await evaluator.verify(
        claim=gen_claim,
        node=gen_verify,
        sources=_pick_sources(info.genre_sources, info.reference_urls),
        additional_instruction="Small naming variations are okay (e.g., 'crime drama' vs 'crime'). Ensure the classification applies to this series."
    )

    # 5) Season number
    sn_node = evaluator.add_sequential(
        id="season_number",
        desc="The answer clearly indicates which season premiered in early 2025",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.season_number),
        id="season_number_provided",
        desc="Season number is provided",
        parent=sn_node,
        critical=True,
    )
    sn_verify = evaluator.add_leaf(
        id="season_number_verified",
        desc="The season number for the early-2025 premiere is correct",
        parent=sn_node,
        critical=True,
    )
    sn_claim = f"The season that premiered in early 2025 is {info.season_number}."
    await evaluator.verify(
        claim=sn_claim,
        node=sn_verify,
        sources=_pick_sources(info.season_sources, info.reference_urls),
        additional_instruction="If the answer calls it 'Season 1' or 'first season' etc., treat them as equivalent."
    )

    # 6) Lead cast
    cast_node = evaluator.add_sequential(
        id="lead_cast",
        desc="The answer identifies at least one lead cast member who stars in the series",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info.lead_cast),
        id="lead_cast_provided",
        desc="At least one lead cast member is provided",
        parent=cast_node,
        critical=True,
    )
    cast_verify = evaluator.add_leaf(
        id="lead_cast_verified",
        desc="The named lead cast member indeed stars in the series",
        parent=cast_node,
        critical=True,
    )
    lead_name = _first_or_empty(info.lead_cast)
    cast_claim = f"{lead_name} is a lead cast member who stars in the series."
    await evaluator.verify(
        claim=cast_claim,
        node=cast_verify,
        sources=_pick_sources(info.cast_sources, info.reference_urls),
        additional_instruction="Accept reasonable spelling or formatting variants of names."
    )

    # 7) Creator / showrunner / EP
    cse_node = evaluator.add_sequential(
        id="creator_producer",
        desc="The answer identifies the creator, showrunner, or executive producer of the series",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.creator_or_showrunner_or_ep),
        id="creator_producer_provided",
        desc="Creator/Showrunner/Executive Producer is provided",
        parent=cse_node,
        critical=True,
    )
    cse_verify = evaluator.add_leaf(
        id="creator_producer_verified",
        desc="The identified creator/showrunner/EP credit is correct",
        parent=cse_node,
        critical=True,
    )
    cse_claim = f"The series' creator, showrunner, or executive producer includes {info.creator_or_showrunner_or_ep}."
    await evaluator.verify(
        claim=cse_claim,
        node=cse_verify,
        sources=_pick_sources(info.creator_sources, info.reference_urls),
        additional_instruction="It is sufficient if the person is a creator OR showrunner OR executive producer as stated."
    )

    # 8) Filming location
    film_node = evaluator.add_sequential(
        id="filming_location",
        desc="If the series involves real-world location filming, the answer specifies the primary filming location",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.filming_location),
        id="filming_location_provided",
        desc="Primary filming location is provided",
        parent=film_node,
        critical=True,
    )
    film_verify = evaluator.add_leaf(
        id="filming_location_verified",
        desc="Primary filming location is correct for this series",
        parent=film_node,
        critical=True,
    )
    film_claim = f"The primary real-world filming location for the series includes: {info.filming_location}."
    await evaluator.verify(
        claim=film_claim,
        node=film_verify,
        sources=_pick_sources(info.filming_sources, info.reference_urls),
        additional_instruction="If multiple locations are listed, verify the one described as primary or principal photography location."
    )

    # 9) Release format
    rel_node = evaluator.add_sequential(
        id="release_format",
        desc="The answer indicates whether episodes are released weekly or all-at-once",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.release_format),
        id="release_format_provided",
        desc="Release model (weekly or all-at-once) is provided",
        parent=rel_node,
        critical=True,
    )
    rel_verify = evaluator.add_leaf(
        id="release_format_verified",
        desc="Release model is correctly stated",
        parent=rel_node,
        critical=True,
    )
    rel_claim = f"The episodes were released {info.release_format}."
    await evaluator.verify(
        claim=rel_claim,
        node=rel_verify,
        sources=_pick_sources(info.release_sources, info.reference_urls),
        additional_instruction="Interpret 'weekly' to include multi-episode premiere followed by weekly rollout. Interpret 'all-at-once' as a full-season drop on one date."
    )

    # 10) Renewal status
    ren_node = evaluator.add_sequential(
        id="renewal_status",
        desc="The answer provides information about renewal status (renewed, cancelled, or pending)",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.renewal_status),
        id="renewal_status_provided",
        desc="Renewal status is provided",
        parent=ren_node,
        critical=True,
    )
    ren_verify = evaluator.add_leaf(
        id="renewal_status_verified",
        desc="Renewal status is correct based on reliable sources",
        parent=ren_node,
        critical=True,
    )
    ren_claim = f"The current renewal status of the series is: {info.renewal_status}."
    await evaluator.verify(
        claim=ren_claim,
        node=ren_verify,
        sources=_pick_sources(info.renewal_sources, info.reference_urls),
        additional_instruction="Use the provided sources to confirm whether it's renewed, cancelled, or pending (or similar clearly-stated status)."
    )

    # 11) Series type (scripted drama/narrative)
    st_node = evaluator.add_sequential(
        id="series_type",
        desc="The series is a scripted drama or narrative series, not reality TV, documentary, or talk show",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.series_type),
        id="series_type_provided",
        desc="Series type is provided",
        parent=st_node,
        critical=True,
    )
    st_verify = evaluator.add_leaf(
        id="series_type_verified",
        desc="Series is scripted drama/narrative (not unscripted/reality/doc/talk)",
        parent=st_node,
        critical=True,
    )
    st_claim = f"This series is a scripted drama or narrative television series (not reality, documentary, or talk), as indicated by its type/genre: {info.series_type}."
    await evaluator.verify(
        claim=st_claim,
        node=st_verify,
        sources=_pick_sources(info.series_type_sources, info.reference_urls),
        additional_instruction="Rely on the provided sources to confirm it is scripted/narrative rather than reality, documentary, game, or talk format."
    )

    # 12) Episode runtime
    rt_node = evaluator.add_sequential(
        id="episode_runtime",
        desc="The answer provides the approximate runtime per episode in minutes",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(info.episode_runtime),
        id="episode_runtime_provided",
        desc="Approximate episode runtime is provided",
        parent=rt_node,
        critical=True,
    )
    rt_verify = evaluator.add_leaf(
        id="episode_runtime_verified",
        desc="Approximate per-episode runtime is correct",
        parent=rt_node,
        critical=True,
    )
    rt_claim = f"The approximate runtime per episode is around {info.episode_runtime} minutes."
    await evaluator.verify(
        claim=rt_claim,
        node=rt_verify,
        sources=_pick_sources(info.runtime_sources, info.reference_urls),
        additional_instruction="Allow ranges (e.g., 45–50) or approximate wording like 'about 60 minutes'."
    )

    # 13) Reference URLs presence and credibility
    ref_node = evaluator.add_parallel(
        id="reference_urls",
        desc="The answer includes reference URLs from the streaming platform or reliable entertainment news sources",
        parent=root_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(info.reference_urls),
        id="reference_urls_provided",
        desc="At least one reference URL is provided in the answer",
        parent=ref_node,
        critical=True,
    )
    ref_cred = evaluator.add_leaf(
        id="reference_urls_credible",
        desc="Provided reference URLs are from streaming platforms or reliable entertainment news sources",
        parent=ref_node,
        critical=True,
    )
    # Use simple verification for credibility judgment (no specific URL evidence is required here)
    listed_urls_str = ", ".join(info.reference_urls) if info.reference_urls else "(none)"
    cred_claim = f"The following URLs are from the streaming platform or reliable entertainment news sources: {listed_urls_str}"
    await evaluator.verify(
        claim=cred_claim,
        node=ref_cred,
        sources=None,
        additional_instruction="Judge credibility by domain: acceptable examples include netflix.com, hulu.com, tv.apple.com or apple.com/tv, max.com/hbo.com, and reputable outlets like variety.com, hollywoodreporter.com, deadline.com, ew.com, indiewire.com, tvline.com, thewrap.com, vulture.com, collider.com, ign.com, screenrant.com, nytimes.com/arts. Minor variants or localized domains are acceptable."
    )

    # Add diagnostic info about sources
    source_stats = {
        "premiere_sources": len(info.premiere_sources),
        "platform_sources": len(info.platform_sources),
        "episode_count_sources": len(info.episode_count_sources),
        "genre_sources": len(info.genre_sources),
        "season_sources": len(info.season_sources),
        "cast_sources": len(info.cast_sources),
        "creator_sources": len(info.creator_sources),
        "filming_sources": len(info.filming_sources),
        "release_sources": len(info.release_sources),
        "renewal_sources": len(info.renewal_sources),
        "series_type_sources": len(info.series_type_sources),
        "runtime_sources": len(info.runtime_sources),
        "reference_urls_total": len(info.reference_urls),
    }
    evaluator.add_custom_info(source_stats, info_type="source_stats")


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
    Evaluate an answer for the 'early2025_scripted_series' task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks per criterion
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

    # 1) Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_series_info(),
        template_class=SeriesInfo,
        extraction_name="series_info",
    )

    # 2) Build verification tree and run checks
    await verify_series_info(evaluator, root, extracted)

    # 3) Return evaluation summary
    return evaluator.get_summary()