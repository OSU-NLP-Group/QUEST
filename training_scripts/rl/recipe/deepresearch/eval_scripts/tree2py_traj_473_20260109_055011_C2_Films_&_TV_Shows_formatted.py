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
TASK_ID = "tv_emmy_2025_drama_s2jan10"
TASK_DESCRIPTION = (
    "Identify a television series that was nominated for Outstanding Drama Series at the 2025 Primetime Emmy Awards "
    "and whose second season premiered in January 2025 with exactly 10 episodes. For this series, provide the following information: "
    "1) The name of the series and the streaming platform where it is available; "
    "2) The creator's full name and confirmation that they serve as an executive producer on the series, with a reference URL supporting this information; "
    "3) At least one production company involved in producing the series, with a reference URL supporting this information; "
    "4) At least one specific building or facility (identified by name and location) that was used as a primary filming location for the series, with a reference URL supporting this information. "
    "All information must be supported by reference URLs from reliable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CreatorInfo(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ProductionCompanyInfo(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class FilmingLocationInfo(BaseModel):
    building_name: Optional[str] = None
    location: Optional[str] = None  # e.g., city, state/country, neighborhood
    reference_urls: List[str] = Field(default_factory=list)


class SeriesSelection(BaseModel):
    series_name: Optional[str] = None
    streaming_platform: Optional[str] = None

    platform_urls: List[str] = Field(default_factory=list)

    emmy_nomination_urls: List[str] = Field(default_factory=list)
    season2_premiere_urls: List[str] = Field(default_factory=list)
    season2_episode_count_urls: List[str] = Field(default_factory=list)

    creator: Optional[CreatorInfo] = None
    production_company: Optional[ProductionCompanyInfo] = None
    filming_location: Optional[FilmingLocationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_selection() -> str:
    return """
    From the provided answer, extract exactly one television series that the answer claims satisfies:
    - It was nominated for Outstanding Drama Series at the 2025 Primetime Emmy Awards.
    - Its second season premiered in January 2025 and has exactly 10 episodes.

    Extract the following fields as a single JSON object:

    1) series_name: The exact series title mentioned.
    2) streaming_platform: The named streaming platform where the series is available (e.g., Netflix, Hulu, Max, Apple TV+).
    3) platform_urls: An array of URLs that specifically support the streaming availability for the series on the named platform.
    4) emmy_nomination_urls: An array of URLs that support the claim that the series was nominated for Outstanding Drama Series at the 2025 Primetime Emmy Awards. Accept official Emmy pages or reliable press coverage.
    5) season2_premiere_urls: An array of URLs that support that the series' season 2 premiered in January 2025. Sources can be reliable trades, official press releases, or the platform/series' official pages.
    6) season2_episode_count_urls: An array of URLs that support that the second season has exactly 10 episodes.
    7) creator: An object with:
       - name: The creator’s full name as stated in the answer.
       - reference_urls: URLs that support both that this person is the creator and that they serve as an executive producer on the series.
    8) production_company: An object with:
       - name: The name of at least one production company involved in producing the series.
       - reference_urls: URLs that support the involvement of this company in the series.
    9) filming_location: An object with:
       - building_name: The name of a specific building or facility used as a primary filming location.
       - location: The location of that building/facility (e.g., city and state/country).
       - reference_urls: URLs that support that this building/facility was a primary (or principal) filming location.

    RULES:
    - Only extract information explicitly mentioned in the answer.
    - For any field not present in the answer, return null or an empty list as appropriate.
    - For URL fields, extract actual URLs that appear in the answer (plain URLs or in markdown). Do not invent URLs.
    - Do not include more than one series; if multiple candidates are present, choose the first one the answer actually uses for the requested details.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification trees                                                          #
# --------------------------------------------------------------------------- #
async def build_eligibility_and_identification(
    evaluator: Evaluator,
    parent,
    data: SeriesSelection,
) -> None:
    """
    Build and verify the 'eligibility_and_identification' subtree.
    """
    node = evaluator.add_parallel(
        id="eligibility_and_identification",
        desc="The selected series is eligible and the answer clearly identifies it and where to watch it, with supporting references.",
        parent=parent,
        critical=True,
    )

    # Basic presence checks
    evaluator.add_custom_node(
        result=bool(data.series_name and data.series_name.strip()),
        id="series_name_provided",
        desc="The answer explicitly states the name of the television series.",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(data.streaming_platform and data.streaming_platform.strip()),
        id="streaming_platform_provided",
        desc="The answer explicitly states a named streaming platform where the series is available.",
        parent=node,
        critical=True,
    )

    # References node – ensure URLs are present and streaming availability is supported by a URL
    refs = evaluator.add_parallel(
        id="references_for_identification_and_eligibility",
        desc="Reference URL(s) are provided that support: (a) the series identification and streaming availability, and (b) the Emmy nomination and season-2 (premiere month + 10-episode) claims.",
        parent=node,
        critical=True,
    )

    # (a) Platform/availability references presence + verification
    evaluator.add_custom_node(
        result=_non_empty_urls(data.platform_urls),
        id="platform_sources_present",
        desc="Reference URL(s) for streaming availability are provided.",
        parent=refs,
        critical=True,
    )
    # Verify streaming availability via provided URLs
    platform_verify_leaf = evaluator.add_leaf(
        id="platform_availability_supported",
        desc="The series is available on the stated streaming platform, supported by the provided URL(s).",
        parent=refs,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The series '{data.series_name or ''}' is available on the streaming platform '{data.streaming_platform or ''}'.",
        node=platform_verify_leaf,
        sources=data.platform_urls if _non_empty_urls(data.platform_urls) else None,
        additional_instruction="Verify that the page(s) show the series is available to stream on the named platform; region-limited availability is acceptable.",
    )

    # (b) Emmy nomination and Season 2 claims – ensure URL presence
    evaluator.add_custom_node(
        result=_non_empty_urls(data.emmy_nomination_urls),
        id="emmy_sources_present",
        desc="Reference URL(s) supporting the Emmy nomination claim are provided.",
        parent=refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(data.season2_premiere_urls),
        id="season2_premiere_sources_present",
        desc="Reference URL(s) supporting the Season 2 January 2025 premiere claim are provided.",
        parent=refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(data.season2_episode_count_urls),
        id="season2_episode_sources_present",
        desc="Reference URL(s) supporting the Season 2 exact 10-episode count claim are provided.",
        parent=refs,
        critical=True,
    )

    # Now verify individual eligibility claims using the provided URLs
    emmy_leaf = evaluator.add_leaf(
        id="emmy_nomination_claim",
        desc="The series was nominated for Outstanding Drama Series at the 2025 Primetime Emmy Awards.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The series '{data.series_name or ''}' was nominated for Outstanding Drama Series at the 2025 Primetime Emmy Awards.",
        node=emmy_leaf,
        sources=data.emmy_nomination_urls if _non_empty_urls(data.emmy_nomination_urls) else None,
        additional_instruction="Prefer references from the Emmys official site or reliable outlets confirming the 2025 Outstanding Drama Series nomination.",
    )

    premiere_leaf = evaluator.add_leaf(
        id="season2_premiere_month_claim",
        desc="The series' second season premiered in January 2025.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The second season of '{data.series_name or ''}' premiered in January 2025.",
        node=premiere_leaf,
        sources=data.season2_premiere_urls if _non_empty_urls(data.season2_premiere_urls) else None,
        additional_instruction="Allow phrasing such as 'Season 2 debuted in January 2025' or 'Season 2 premiered January 2025'.",
    )

    eps_leaf = evaluator.add_leaf(
        id="season2_episode_count_claim",
        desc="The series' second season contains exactly 10 episodes.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The second season of '{data.series_name or ''}' has exactly 10 episodes.",
        node=eps_leaf,
        sources=data.season2_episode_count_urls if _non_empty_urls(data.season2_episode_count_urls) else None,
        additional_instruction="Confirm the total episode count for Season 2 is exactly 10; do not infer from other seasons.",
    )


async def build_requested_details(
    evaluator: Evaluator,
    parent,
    data: SeriesSelection,
) -> None:
    """
    Build and verify the 'requested_details_with_references' subtree.
    """
    details_node = evaluator.add_parallel(
        id="requested_details_with_references",
        desc="All requested details about creator, production company, and filming location are provided, each supported by reference URL(s).",
        parent=parent,
        critical=True,
    )

    # Creator
    creator_root = evaluator.add_parallel(
        id="creator_name_and_ep_with_reference",
        desc="Provides the creator’s full name, confirms the creator serves as an executive producer, and includes a reference URL supporting this.",
        parent=details_node,
        critical=True,
    )
    creator_name_provided = evaluator.add_custom_node(
        result=bool(data.creator and data.creator.name and data.creator.name.strip()),
        id="creator_name_provided",
        desc="Creator full name is provided.",
        parent=creator_root,
        critical=True,
    )
    creator_refs_present = evaluator.add_custom_node(
        result=bool(data.creator and _non_empty_urls(data.creator.reference_urls)),
        id="creator_sources_present",
        desc="Reference URL(s) for creator and EP role are provided.",
        parent=creator_root,
        critical=True,
    )
    # Verify creator role
    creator_is_creator_leaf = evaluator.add_leaf(
        id="creator_is_creator_supported",
        desc="The provided person is the creator of the series, supported by the reference URL(s).",
        parent=creator_root,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{(data.creator.name if data.creator else '')}' is the creator of the series '{data.series_name or ''}'.",
        node=creator_is_creator_leaf,
        sources=(data.creator.reference_urls if (data.creator and _non_empty_urls(data.creator.reference_urls)) else None),
        additional_instruction="Accept equivalent titles like 'created by'; verify that the referenced page explicitly names this person as the series creator.",
    )
    creator_is_ep_leaf = evaluator.add_leaf(
        id="creator_is_ep_supported",
        desc="The creator also serves as an executive producer on the series, supported by the reference URL(s).",
        parent=creator_root,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{(data.creator.name if data.creator else '')}' serves as an executive producer on the series '{data.series_name or ''}'.",
        node=creator_is_ep_leaf,
        sources=(data.creator.reference_urls if (data.creator and _non_empty_urls(data.creator.reference_urls)) else None),
        additional_instruction="Look for explicit mention of 'executive producer' in relation to the creator on the page.",
    )

    # Production company
    prod_root = evaluator.add_parallel(
        id="production_company_with_reference",
        desc="Names at least one production company involved in producing the series and includes a reference URL supporting this.",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(data.production_company and data.production_company.name and data.production_company.name.strip()),
        id="production_company_provided",
        desc="A production company name is provided.",
        parent=prod_root,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(data.production_company and _non_empty_urls(data.production_company.reference_urls)),
        id="production_company_sources_present",
        desc="Reference URL(s) for the production company involvement are provided.",
        parent=prod_root,
        critical=True,
    )
    prod_supported_leaf = evaluator.add_leaf(
        id="production_company_supported",
        desc="The named production company is involved in producing the series, supported by the reference URL(s).",
        parent=prod_root,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{(data.production_company.name if data.production_company else '')}' is one of the production companies involved in producing the series '{data.series_name or ''}'.",
        node=prod_supported_leaf,
        sources=(data.production_company.reference_urls if (data.production_company and _non_empty_urls(data.production_company.reference_urls)) else None),
        additional_instruction="Verify that the page attributes the company as a producer/production company for the series.",
    )

    # Filming location (building/facility)
    film_root = evaluator.add_parallel(
        id="filming_location_building_with_reference",
        desc="Identifies at least one specific building or facility used as a primary filming location (by name and location) and includes a reference URL supporting this.",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(data.filming_location and data.filming_location.building_name and data.filming_location.building_name.strip()),
        id="filming_building_name_provided",
        desc="A building/facility name is provided as a filming location.",
        parent=film_root,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(data.filming_location and data.filming_location.location and data.filming_location.location.strip()),
        id="filming_building_location_provided",
        desc="A location (city/state/country) is provided for the filming building/facility.",
        parent=film_root,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(data.filming_location and _non_empty_urls(data.filming_location.reference_urls)),
        id="filming_location_sources_present",
        desc="Reference URL(s) for the building/facility filming location are provided.",
        parent=film_root,
        critical=True,
    )
    filming_supported_leaf = evaluator.add_leaf(
        id="filming_building_supported",
        desc="The named building/facility at the specified location was a primary filming location for the series, supported by the reference URL(s).",
        parent=film_root,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The building/facility '{(data.filming_location.building_name if data.filming_location else '')}' "
            f"in {(data.filming_location.location if data.filming_location else '')} was used as a primary filming location for "
            f"the series '{data.series_name or ''}'."
        ),
        node=filming_supported_leaf,
        sources=(data.filming_location.reference_urls if (data.filming_location and _non_empty_urls(data.filming_location.reference_urls)) else None),
        additional_instruction="Accept phrasing such as principal photography at, primary filming took place at, or repeatedly used as a main filming site.",
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
    """
    Entry point for evaluating an answer for the TV series eligibility and details task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Framework root is non-critical; we add a critical task node under it.
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

    # Extract structured information from the answer
    series_data: SeriesSelection = await evaluator.extract(
        prompt=prompt_extract_series_selection(),
        template_class=SeriesSelection,
        extraction_name="series_selection",
    )

    # Create a critical sequential node to reflect the rubric root
    task_root = evaluator.add_sequential(
        id="task_root",
        desc="Identify a TV drama series that satisfies the Emmy nomination and season constraints, then provide the requested series/creator/production/filming-location details, with supporting reference URLs.",
        parent=root,
        critical=True,
    )

    # 1) Eligibility and Identification
    await build_eligibility_and_identification(evaluator, task_root, series_data)

    # 2) Requested Details with References
    await build_requested_details(evaluator, task_root, series_data)

    return evaluator.get_summary()