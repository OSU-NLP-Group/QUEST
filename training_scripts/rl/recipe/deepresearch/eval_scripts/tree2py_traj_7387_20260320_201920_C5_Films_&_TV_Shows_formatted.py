import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "animated_series_6seasons_podcast_company_2020_2014"
TASK_DESCRIPTION = """
Identify an animated television series that meets all of the following criteria:

1. The series had exactly 6 seasons and was exclusively available on a major streaming platform as an original series.
2. The lead voice actor of the series also co-hosts a podcast that launched in 2020 with two other co-hosts.
3. The same voice actor founded a production company in 2014.

In your answer, provide:
- The name of the animated series
- The name of the lead voice actor
- The name of the streaming platform
- The name of the production company founded by the voice actor
- The name of the podcast co-hosted by the voice actor
- URL references supporting each piece of information
""".strip()


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SeriesInfo(BaseModel):
    series_name: Optional[str] = None
    # URLs explicitly cited supporting that the series is an animated television series
    series_animated_urls: List[str] = Field(default_factory=list)

    # Seasons info (as stated in the answer) and URLs that support season count
    seasons_count: Optional[str] = None
    seasons_urls: List[str] = Field(default_factory=list)

    # Streaming platform name and URLs supporting "platform original/exclusive"
    platform_name: Optional[str] = None
    platform_original_exclusive_urls: List[str] = Field(default_factory=list)

    # URLs supporting that the platform is a "major" streaming platform/service
    platform_major_urls: List[str] = Field(default_factory=list)


class VoiceActorInfo(BaseModel):
    # Lead voice actor for the series and supporting URLs
    lead_voice_actor: Optional[str] = None
    lead_voice_actor_urls: List[str] = Field(default_factory=list)

    # Production company founded by the voice actor and supporting URLs
    production_company_name: Optional[str] = None
    production_company_urls: List[str] = Field(default_factory=list)
    production_company_founded_year_stated: Optional[str] = None  # as stated in the answer (if any)

    # Podcast info
    podcast_name: Optional[str] = None
    # URLs that support co-hosting and/or podcast general info
    podcast_urls: List[str] = Field(default_factory=list)

    # Podcast launch year as stated in the answer (if any) and URLs supporting it
    podcast_launch_year_stated: Optional[str] = None
    podcast_launch_urls: List[str] = Field(default_factory=list)

    # Podcast co-hosts as stated in the answer (if any) and URLs supporting the co-host lineup/count
    podcast_cohosts_stated: List[str] = Field(default_factory=list)
    podcast_hostcount_urls: List[str] = Field(default_factory=list)


class TaskExtraction(BaseModel):
    series: Optional[SeriesInfo] = None
    voice_actor: Optional[VoiceActorInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return """
Extract the following fields exactly from the provided answer. Do not invent or infer anything not explicitly present in the answer text. For every URL field, include only explicit URLs that appear in the answer (plain URLs or markdown links). If a field is missing, set it to null (for strings) or [] (for lists).

Return a JSON object with this structure:

{
  "series": {
    "series_name": string|null,
    "series_animated_urls": string[],

    "seasons_count": string|null,
    "seasons_urls": string[],

    "platform_name": string|null,
    "platform_original_exclusive_urls": string[],
    "platform_major_urls": string[]
  },
  "voice_actor": {
    "lead_voice_actor": string|null,
    "lead_voice_actor_urls": string[],

    "production_company_name": string|null,
    "production_company_urls": string[],
    "production_company_founded_year_stated": string|null,

    "podcast_name": string|null,
    "podcast_urls": string[],

    "podcast_launch_year_stated": string|null,
    "podcast_launch_urls": string[],

    "podcast_cohosts_stated": string[],
    "podcast_hostcount_urls": string[]
  }
}

Field-specific guidance:
- series.series_animated_urls: URLs the answer cites to support that the series is an "animated television series" (accept synonyms like "animated series", "animated sitcom", "adult animated series", etc.).
- series.seasons_count: Use the exact text the answer uses for the season count (e.g., "6", "six", "exactly six seasons", "ran for 6 seasons").
- series.seasons_urls: URLs the answer cites that support the 6-season count.
- series.platform_name: The streaming platform named in the answer (e.g., Netflix, Hulu, Amazon Prime Video, etc.).
- series.platform_original_exclusive_urls: URLs the answer cites to support that the series was a "{platform_name} original" or otherwise exclusive to that platform.
- series.platform_major_urls: URLs the answer cites showing the platform is a "major" streaming platform/service (articles, reputable pages, etc., explicitly describing it as major/top).

- voice_actor.lead_voice_actor: The lead voice actor name the answer states.
- voice_actor.lead_voice_actor_urls: URLs supporting that this person is the lead voice actor of the series.
- voice_actor.production_company_name: The name of a production company the answer says the voice actor founded.
- voice_actor.production_company_urls: URLs supporting the founding and/or details of the company.
- voice_actor.production_company_founded_year_stated: The founding year the answer states for the company (if any).
- voice_actor.podcast_name: The podcast name the answer states.
- voice_actor.podcast_urls: URLs supporting that the person co-hosts the named podcast (general podcast info also OK).
- voice_actor.podcast_launch_year_stated: The launch year the answer states for the podcast (if any).
- voice_actor.podcast_launch_urls: URLs the answer cites that support the launch year.
- voice_actor.podcast_cohosts_stated: The list of co-host names as stated in the answer (if mentioned).
- voice_actor.podcast_hostcount_urls: URLs supporting the total number of co-hosts / lineup.

Rules:
- For URL lists, include every explicit URL that the answer associates with the specific fact. Do not include any URL that is not clearly connected to that fact in the answer.
- If the answer provides no URL for a particular field, return an empty list for that field.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10
}


def normalize_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    s = text.strip().lower()
    # Try digits first
    m = re.search(r"\b(\d+)\b", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Try word numbers
    for w, n in _WORD_TO_NUM.items():
        if re.search(rf"\b{re.escape(w)}\b", s):
            return n
    return None


def non_empty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_series_requirements(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    """
    Build the "series_requirements" subtree (critical, parallel).
    This includes:
      - Series name + animated proof
      - Exactly 6 seasons + supporting URL
      - Platform original/exclusive + platform is major (both supported)
    """
    series = data.series or SeriesInfo()

    series_root = evaluator.add_parallel(
        id="series_requirements",
        desc="Series information satisfies all series-side constraints and is properly sourced.",
        parent=parent_node,
        critical=True
    )

    # 1) Series name + animated proof
    name_anim_node = evaluator.add_parallel(
        id="series_name_and_animated_with_source",
        desc="Provides the series name AND at least one URL that supports that it is an animated television series.",
        parent=series_root,
        critical=True
    )

    name_present_node = evaluator.add_custom_node(
        result=bool(series.series_name and series.series_name.strip()) and non_empty_urls(series.series_animated_urls),
        id="series_name_present_and_has_animated_url",
        desc="Series name provided and at least one 'animated' supporting URL is present in the answer.",
        parent=name_anim_node,
        critical=True
    )

    animated_leaf = evaluator.add_leaf(
        id="series_is_animated_supported_by_urls",
        desc="Cited URLs support that the identified series is an animated television series.",
        parent=name_anim_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{series.series_name or ''}' is an animated television series (a.k.a. animated series/animated sitcom/adult animated).",
        node=animated_leaf,
        sources=series.series_animated_urls,
        additional_instruction="Accept synonyms like 'animated series', 'animated sitcom', 'adult animated series'. The URL must clearly indicate the series is an animated TV series."
    )

    # 2) Exactly 6 seasons + supporting URL(s)
    seasons_node = evaluator.add_parallel(
        id="exactly_6_seasons_with_source",
        desc="States that the series has exactly 6 seasons AND provides at least one URL supporting the 6-season count.",
        parent=series_root,
        critical=True
    )

    stated_count = normalize_int_from_text(series.seasons_count)
    stated_six = stated_count == 6
    seasons_stated_leaf = evaluator.add_custom_node(
        result=stated_six,
        id="series_seasons_stated_exactly_six",
        desc="Answer explicitly states the series has exactly 6 seasons.",
        parent=seasons_node,
        critical=True
    )

    seasons_url_present_leaf = evaluator.add_custom_node(
        result=non_empty_urls(series.seasons_urls),
        id="series_seasons_has_supporting_url",
        desc="At least one URL is provided to support the 6-season count.",
        parent=seasons_node,
        critical=True
    )

    seasons_supported_leaf = evaluator.add_leaf(
        id="series_seasons_supported_by_urls",
        desc="Cited URLs support that the series has exactly 6 seasons.",
        parent=seasons_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{series.series_name or ''}' has exactly 6 seasons.",
        node=seasons_supported_leaf,
        sources=series.seasons_urls,
        additional_instruction="Treat phrases like 'ran for six seasons', '6 seasons total', or equivalent wording as valid support for exactly six seasons."
    )

    # 3) Platform constraints: original/exclusive + platform is major
    platform_node = evaluator.add_parallel(
        id="platform_major_original_exclusive_with_source",
        desc="Names the streaming platform AND provides URL(s) supporting (a) the series was a platform original/exclusive AND (b) the platform is described as a major streaming platform/service.",
        parent=series_root,
        critical=True
    )

    platform_name_present = evaluator.add_custom_node(
        result=bool(series.platform_name and series.platform_name.strip()),
        id="platform_name_present",
        desc="The streaming platform name is provided.",
        parent=platform_node,
        critical=True
    )

    original_urls_present = evaluator.add_custom_node(
        result=non_empty_urls(series.platform_original_exclusive_urls),
        id="platform_original_exclusive_url_present",
        desc="At least one URL is provided to support 'original/exclusive to the platform'.",
        parent=platform_node,
        critical=True
    )

    major_urls_present = evaluator.add_custom_node(
        result=non_empty_urls(series.platform_major_urls),
        id="platform_major_url_present",
        desc="At least one URL is provided to support the platform being 'major'.",
        parent=platform_node,
        critical=True
    )

    original_exclusive_leaf = evaluator.add_leaf(
        id="platform_original_exclusive_supported",
        desc="Cited URLs support that the series was a platform original/exclusive.",
        parent=platform_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The series '{series.series_name or ''}' was a {series.platform_name or ''} original series or otherwise exclusive to {series.platform_name or ''}.",
        node=original_exclusive_leaf,
        sources=series.platform_original_exclusive_urls,
        additional_instruction="Accept variations like 'original series', 'exclusive original', or the platform page labeling the series as an 'Original'. The URL must associate the series with the stated platform."
    )

    platform_major_leaf = evaluator.add_leaf(
        id="platform_is_major_supported",
        desc="Cited URLs support that the named platform is a 'major' streaming platform/service.",
        parent=platform_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{series.platform_name or ''} is a major streaming platform/service.",
        node=platform_major_leaf,
        sources=series.platform_major_urls,
        additional_instruction="Support may include wording like 'major streaming service', 'one of the major streamers', 'top streaming platform', or other credible phrasing indicating it is a major/leading streaming service."
    )


async def build_lead_voice_actor_requirements(evaluator: Evaluator, parent_node, data: TaskExtraction) -> None:
    """
    Build the "lead_voice_actor_requirements" subtree (critical, parallel).
    This includes:
      - Lead voice actor + supporting URL(s)
      - Production company founded by that actor in 2014 (supported)
      - Podcast name + actor co-hosting (supported)
      - Podcast launched in 2020 (stated + supported)
      - Podcast has exactly three co-hosts (stated + supported)
      - Identity consistency across actor ↔ podcast host ↔ company founder
    """
    actor = data.voice_actor or VoiceActorInfo()
    series = (data.series or SeriesInfo())

    actor_root = evaluator.add_parallel(
        id="lead_voice_actor_requirements",
        desc="Lead voice actor is correctly identified, meets podcast/company constraints, and is properly sourced.",
        parent=parent_node,
        critical=True
    )

    # 1) Lead voice actor + supporting URL(s)
    actor_with_source = evaluator.add_parallel(
        id="lead_voice_actor_with_source",
        desc="Provides the lead voice actor name AND at least one URL supporting they are the lead voice actor for the series.",
        parent=actor_root,
        critical=True
    )

    actor_name_and_url_present = evaluator.add_custom_node(
        result=bool(actor.lead_voice_actor and actor.lead_voice_actor.strip()) and non_empty_urls(actor.lead_voice_actor_urls),
        id="lead_voice_actor_name_and_url_present",
        desc="Lead voice actor name provided and at least one supporting URL present.",
        parent=actor_with_source,
        critical=True
    )

    actor_supported_leaf = evaluator.add_leaf(
        id="lead_voice_actor_role_supported",
        desc="Cited URLs support that this person is the lead voice actor of the identified series.",
        parent=actor_with_source,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{actor.lead_voice_actor or ''}' is the lead voice actor of the series '{series.series_name or ''}'.",
        node=actor_supported_leaf,
        sources=actor.lead_voice_actor_urls,
        additional_instruction="Accept phrasings like 'lead role', 'main voice', 'starring voice', or being credited as the main/lead voice actor. The URL must clearly indicate lead/main status for the series."
    )

    # 2) Production company founded by voice actor in 2014
    company_group = evaluator.add_parallel(
        id="production_company_founded_2014_with_source",
        desc="Provides the production company name AND at least one URL supporting the (lead) voice actor founded it in 2014.",
        parent=actor_root,
        critical=True
    )

    company_name_present = evaluator.add_custom_node(
        result=bool(actor.production_company_name and actor.production_company_name.strip()) and non_empty_urls(actor.production_company_urls),
        id="production_company_name_and_url_present",
        desc="Production company name present and at least one supporting URL provided.",
        parent=company_group,
        critical=True
    )

    company_founded_by_actor_leaf = evaluator.add_leaf(
        id="company_founded_by_actor_supported",
        desc="Cited URLs support that the lead voice actor founded (or co-founded) the named production company.",
        parent=company_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{actor.lead_voice_actor or ''}' founded (or co-founded) the production company '{actor.production_company_name or ''}'.",
        node=company_founded_by_actor_leaf,
        sources=actor.production_company_urls,
        additional_instruction="The URL should clearly indicate the person is the founder or co-founder of the company."
    )

    company_founded_in_2014_leaf = evaluator.add_leaf(
        id="company_founded_in_2014_supported",
        desc="Cited URLs support that the production company was founded in 2014.",
        parent=company_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The production company '{actor.production_company_name or ''}' was founded in 2014.",
        node=company_founded_in_2014_leaf,
        sources=actor.production_company_urls,
        additional_instruction="Accept synonyms like 'founded in 2014', 'established in 2014', or equivalent wording."
    )

    # 3) Podcast name + co-hosting supported
    podcast_group = evaluator.add_parallel(
        id="podcast_name_and_cohosting_with_source",
        desc="Provides the podcast name AND at least one URL supporting the (lead) voice actor is a co-host of the podcast.",
        parent=actor_root,
        critical=True
    )

    podcast_name_present = evaluator.add_custom_node(
        result=bool(actor.podcast_name and actor.podcast_name.strip()) and non_empty_urls(actor.podcast_urls),
        id="podcast_name_and_url_present",
        desc="Podcast name present and at least one supporting URL provided.",
        parent=podcast_group,
        critical=True
    )

    podcast_cohost_supported_leaf = evaluator.add_leaf(
        id="podcast_cohost_role_supported",
        desc="Cited URLs support that the lead voice actor is a co-host of the named podcast.",
        parent=podcast_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{actor.lead_voice_actor or ''}' is a co-host of the podcast '{actor.podcast_name or ''}'.",
        node=podcast_cohost_supported_leaf,
        sources=actor.podcast_urls,
        additional_instruction="Accept synonyms like 'host', 'co-host', or 'presenter'. The URL should show the person as part of the regular hosting lineup."
    )

    # 4) Podcast launched in 2020 (stated + supported)
    podcast_launch_node = evaluator.add_parallel(
        id="podcast_launched_2020_with_source",
        desc="States the podcast launched in 2020 AND provides at least one URL supporting the 2020 launch date.",
        parent=actor_root,
        critical=True
    )

    stated_launch_year = normalize_int_from_text(actor.podcast_launch_year_stated)
    stated_2020 = stated_launch_year == 2020
    podcast_launch_stated_leaf = evaluator.add_custom_node(
        result=stated_2020,
        id="podcast_launch_year_stated_2020",
        desc="Answer explicitly states the podcast launched in 2020.",
        parent=podcast_launch_node,
        critical=True
    )

    launch_urls = actor.podcast_launch_urls if non_empty_urls(actor.podcast_launch_urls) else actor.podcast_urls
    launch_url_present_leaf = evaluator.add_custom_node(
        result=non_empty_urls(launch_urls),
        id="podcast_launch_url_present",
        desc="At least one URL is provided to support the 2020 launch year.",
        parent=podcast_launch_node,
        critical=True
    )

    podcast_launch_supported_leaf = evaluator.add_leaf(
        id="podcast_launch_2020_supported",
        desc="Cited URLs support that the podcast launched in 2020.",
        parent=podcast_launch_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The podcast '{actor.podcast_name or ''}' launched in 2020.",
        node=podcast_launch_supported_leaf,
        sources=launch_urls,
        additional_instruction="Accept wording like 'launched in 2020', 'debuted in 2020', 'premiered in 2020'."
    )

    # 5) Podcast has exactly 3 co-hosts (stated + supported)
    podcast_hosts_node = evaluator.add_parallel(
        id="podcast_three_cohosts_with_source",
        desc="States the podcast has exactly 3 co-hosts total (the voice actor plus two others) AND provides at least one URL supporting the co-host count/lineup.",
        parent=actor_root,
        critical=True
    )

    # Determine if the answer stated exactly three co-hosts
    stated_hosts_count = None
    if actor.podcast_cohosts_stated:
        stated_hosts_count = len(actor.podcast_cohosts_stated)
    # If not explicit list, try to parse count from wording (if any appeared elsewhere, not common; fallback to None)
    stated_three = (stated_hosts_count == 3)

    podcast_three_stated_leaf = evaluator.add_custom_node(
        result=stated_three,
        id="podcast_three_cohosts_stated",
        desc="Answer explicitly states or lists exactly three total co-hosts.",
        parent=podcast_hosts_node,
        critical=True
    )

    hostcount_urls = actor.podcast_hostcount_urls if non_empty_urls(actor.podcast_hostcount_urls) else actor.podcast_urls
    hostcount_url_present_leaf = evaluator.add_custom_node(
        result=non_empty_urls(hostcount_urls),
        id="podcast_hostcount_url_present",
        desc="At least one URL is provided to support the three co-hosts total.",
        parent=podcast_hosts_node,
        critical=True
    )

    podcast_three_supported_leaf = evaluator.add_leaf(
        id="podcast_three_cohosts_supported",
        desc="Cited URLs support that the podcast has exactly three co-hosts.",
        parent=podcast_hosts_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The podcast '{actor.podcast_name or ''}' has exactly three co-hosts in total.",
        node=podcast_three_supported_leaf,
        sources=hostcount_urls,
        additional_instruction="Accept wording like 'co-hosted by three people', 'a trio of hosts', or explicit lists that total three regular co-hosts."
    )

    # 6) Same voice actor consistency across roles (identity check; simple verify)
    identity_leaf = evaluator.add_leaf(
        id="same_voice_actor_consistency",
        desc="The person named as lead voice actor is the same person referenced as the podcast co-host and as the production company founder (no mismatched identities).",
        parent=actor_root,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The lead voice actor '{actor.lead_voice_actor or ''}' is the same individual who co-hosts the podcast "
            f"'{actor.podcast_name or ''}' and who founded the production company '{actor.production_company_name or ''}'."
        ),
        node=identity_leaf,
        additional_instruction="Focus on name/entity consistency across the provided answer fields. Allow minor variations (middle initials, hyphenation, capitalization). Do not use external knowledge; use the answer context."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Entry point for evaluating an answer against the rubric for the animated series task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation as parallel per rubric
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

    # Add a critical task root under the evaluator root to respect "critical root" semantics
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Answer identifies one animated TV series meeting all constraints and provides all requested fields with supporting URLs.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=TaskExtraction,
        extraction_name="parsed_answer_fields"
    )

    # Build verification subtrees
    await build_series_requirements(evaluator, task_root, extracted)
    await build_lead_voice_actor_requirements(evaluator, task_root, extracted)

    # Return summary
    return evaluator.get_summary()