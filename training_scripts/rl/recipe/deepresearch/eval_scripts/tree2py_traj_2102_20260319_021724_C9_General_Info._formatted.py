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
TASK_ID = "entertainment_q1_2026"
TASK_DESCRIPTION = """
Identify four distinct entertainment releases or events that took place between January 1, 2026, and March 20, 2026, with each item coming from a different category as specified below. For each item, you must provide: (1) the exact title, (2) the exact release, premiere, or participation date (or date range for events), (3) the platform, venue, or network where it was released or took place, and (4) at least two individuals involved in the production (such as cast members, director, writer, or creator) along with their specific roles.

The four required categories are:
1. A theatrical film that was released in movie theaters during this period
2. A film or series that premiered on a streaming platform (such as Netflix, Prime Video, etc.) during this period
3. A specific episode of a television series that aired during this period on a broadcast or cable network
4. A Broadway show that participated in NYC Broadway Week, which ran from January 20 to February 12, 2026

For the Broadway show, you must verify that the show was officially listed as one of the participating productions offering 2-for-1 tickets during NYC Broadway Week 2026.
"""

WINDOW_START = "January 1, 2026"
WINDOW_END = "March 20, 2026"
BROADWAY_WEEK_START = "January 20, 2026"
BROADWAY_WEEK_END = "February 12, 2026"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PersonRole(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class TheatricalFilmItem(BaseModel):
    title: Optional[str] = None
    release_date: Optional[str] = None
    distributor_or_release_info: Optional[str] = None
    info_urls: List[str] = Field(default_factory=list)
    persons: List[PersonRole] = Field(default_factory=list)


class StreamingItem(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None
    platform: Optional[str] = None
    info_urls: List[str] = Field(default_factory=list)
    persons: List[PersonRole] = Field(default_factory=list)


class TVEpisodeItem(BaseModel):
    series_title: Optional[str] = None
    episode_title: Optional[str] = None
    air_date: Optional[str] = None
    network: Optional[str] = None
    info_urls: List[str] = Field(default_factory=list)
    persons: List[PersonRole] = Field(default_factory=list)


class BroadwayShowItem(BaseModel):
    title: Optional[str] = None
    venue: Optional[str] = None
    participation_urls: List[str] = Field(default_factory=list)
    info_urls: List[str] = Field(default_factory=list)
    persons: List[PersonRole] = Field(default_factory=list)


class EntertainmentExtraction(BaseModel):
    theatrical_film: Optional[TheatricalFilmItem] = None
    streaming_item: Optional[StreamingItem] = None
    tv_episode: Optional[TVEpisodeItem] = None
    broadway_show: Optional[BroadwayShowItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_entertainment_items() -> str:
    return f"""
Extract structured information for FOUR distinct items from the answer, one per required category. Use EXACT strings as they appear in the answer. Return null for any entire item not present.

GENERAL RULES:
- Do not invent or infer missing information.
- Only extract URLs that are explicitly present in the answer.
- For each item's 'persons' list, extract at least two people with their specific roles as written (e.g., "actor as Character", "director", "writer").
- For every person, extract any associated source URLs explicitly tied to that person's involvement. If none are given, return an empty list.
- For info_urls and participation_urls, only include valid URLs explicitly provided.

FIELDS TO EXTRACT:

1) theatrical_film:
- title: exact theatrical film title
- release_date: the theatrical release date string as provided
- distributor_or_release_info: distributor name (e.g., A24, Lionsgate, Universal) OR explicit theatrical release descriptor provided in the answer (e.g., "limited release", "wide release", "in theaters nationwide")
- info_urls: list of URLs that support the film’s basic information (title, distributor/release info, release date); extract all explicit URLs tied to this item
- persons: array of objects:
    - name
    - role (e.g., "actor as X", "director", "writer")
    - source_urls: list of URLs that confirm this person's role/involvement

2) streaming_item:
- title: exact streaming film or series title
- premiere_date: the streaming premiere date string as provided
- platform: the streaming platform (e.g., Netflix, Prime Video, Hulu, Disney+, Max)
- info_urls: list of URLs that support the streaming item's basic info (title, platform, premiere date)
- persons: array of objects:
    - name
    - role
    - source_urls

3) tv_episode:
- series_title: exact series name
- episode_title: exact episode title
- air_date: the episode’s air date string as provided
- network: the broadcast or cable network (e.g., CBS, NBC, ABC, FOX, HBO, FX)
- info_urls: list of URLs that support the TV episode’s basic info (series, episode, air date, network)
- persons: array of objects:
    - name
    - role
    - source_urls

4) broadway_show:
- title: exact Broadway production title
- venue: the Broadway theater venue name or explicit "Broadway" designation as provided
- participation_urls: list of URLs that explicitly show the show participated in NYC Broadway Week 2026 (2-for-1 tickets; event ran {BROADWAY_WEEK_START}–{BROADWAY_WEEK_END}); include only URLs that directly support the participation claim
- info_urls: list of other URLs supporting the show's basic info (title/venue/cast/etc.)
- persons: array of objects:
    - name
    - role
    - source_urls

Return a single JSON object with keys: theatrical_film, streaming_item, tv_episode, broadway_show.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _two_persons(persons: List[PersonRole]) -> List[PersonRole]:
    ppl = (persons or [])[:2]
    while len(ppl) < 2:
        ppl.append(PersonRole())
    return ppl


def _merge_sources(*lists_of_urls: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists_of_urls:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str) and u.strip():
                merged.append(u.strip())
    return merged


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_theatrical_film(evaluator: Evaluator, parent, item: Optional[TheatricalFilmItem]) -> None:
    film_node = evaluator.add_parallel(
        id="theatrical_film_item",
        desc="Evaluate the theatrical film entry",
        parent=parent,
        critical=False
    )

    # Basic info (critical)
    basic = evaluator.add_parallel(
        id="theatrical_film_basic_info",
        desc="Verify basic identifying information for the theatrical film",
        parent=film_node,
        critical=True
    )

    info_urls = item.info_urls if item else []
    title = (item.title or "") if item else ""
    release_date = (item.release_date or "") if item else ""
    dist_info = (item.distributor_or_release_info or "") if item else ""

    # URL presence (critical gating)
    evaluator.add_custom_node(
        result=bool(info_urls),
        id="theatrical_film_basic_info_url",
        desc="URL reference supporting the basic information about the theatrical film",
        parent=basic,
        critical=True
    )

    # Title
    title_leaf = evaluator.add_leaf(
        id="theatrical_film_title",
        desc="The exact title of the theatrical film is provided",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) confirm that the theatrical film's title is exactly '{title}'.",
        node=title_leaf,
        sources=info_urls,
        additional_instruction="Verify the page is about this film and that the title matches (minor case/punctuation differences are acceptable)."
    )

    # Release date within window
    release_leaf = evaluator.add_leaf(
        id="theatrical_film_release_date",
        desc=f"The theatrical release date is provided and falls between {WINDOW_START}, and {WINDOW_END}",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) confirm that the film '{title}' had a theatrical release on {release_date}, and this date falls between {WINDOW_START} and {WINDOW_END}.",
        node=release_leaf,
        sources=info_urls,
        additional_instruction="Confirm a theatrical release date and judge whether it lies within the window. If multiple regions are listed, any theatrical date within the window is acceptable."
    )

    # Distributor / theatrical release info
    platform_leaf = evaluator.add_leaf(
        id="theatrical_film_platform",
        desc="The distribution company or theatrical release information is specified (e.g., A24, Lionsgate, etc.)",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) indicate the theatrical distributor or distribution information for '{title}' as '{dist_info}'.",
        node=platform_leaf,
        sources=info_urls,
        additional_instruction="Accept either an explicit distributor (e.g., A24, Universal) or an explicit theatrical release descriptor that demonstrates it was released in theaters."
    )

    # Personnel (critical)
    personnel_node = evaluator.add_parallel(
        id="theatrical_film_personnel",
        desc="Verify at least two personnel involved in the theatrical film with their roles",
        parent=film_node,
        critical=True
    )

    persons = _two_persons(item.persons if item else [])
    for idx, p in enumerate(persons, start=1):
        # URL existence for person
        evaluator.add_custom_node(
            result=bool(p.source_urls),
            id=f"theatrical_film_person_{idx}_url",
            desc=f"URL reference confirming the {'first' if idx == 1 else 'second'} individual's involvement in the theatrical film",
            parent=personnel_node,
            critical=True
        )

        # Person role verification
        person_leaf = evaluator.add_leaf(
            id=f"theatrical_film_person_{idx}",
            desc=f"{'First' if idx == 1 else 'Second'} individual involved in the theatrical film is identified with their specific role (e.g., actor as character name, director, writer)",
            parent=personnel_node,
            critical=True
        )
        all_sources = _merge_sources(p.source_urls, info_urls)
        await evaluator.verify(
            claim=f"The page(s) confirm that {p.name or ''} served as {p.role or ''} for the theatrical film '{title}'.",
            node=person_leaf,
            sources=all_sources,
            additional_instruction="Confirm both the person's name and the stated role for this film. Accept small wording variations (e.g., 'cast', 'starring' for actors)."
        )


async def verify_streaming_item(evaluator: Evaluator, parent, item: Optional[StreamingItem]) -> None:
    node = evaluator.add_parallel(
        id="streaming_item",
        desc="Evaluate the streaming film or series entry",
        parent=parent,
        critical=False
    )

    basic = evaluator.add_parallel(
        id="streaming_basic_info",
        desc="Verify basic identifying information for the streaming item",
        parent=node,
        critical=True
    )

    info_urls = item.info_urls if item else []
    title = (item.title or "") if item else ""
    premiere_date = (item.premiere_date or "") if item else ""
    platform = (item.platform or "") if item else ""

    evaluator.add_custom_node(
        result=bool(info_urls),
        id="streaming_basic_info_url",
        desc="URL reference supporting the basic information about the streaming item",
        parent=basic,
        critical=True
    )

    title_leaf = evaluator.add_leaf(
        id="streaming_title",
        desc="The exact title of the streaming film or series is provided",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) confirm the streaming title is exactly '{title}'.",
        node=title_leaf,
        sources=info_urls,
        additional_instruction="Verify the page is about this streaming title; allow small punctuation/case differences."
    )

    date_leaf = evaluator.add_leaf(
        id="streaming_premiere_date",
        desc=f"The streaming premiere date is provided and falls between {WINDOW_START}, and {WINDOW_END}",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) confirm that '{title}' premiered on the streaming platform on {premiere_date}, and this date falls between {WINDOW_START} and {WINDOW_END}.",
        node=date_leaf,
        sources=info_urls,
        additional_instruction="Confirm the streaming premiere date (first release on the platform) and judge it within the window."
    )

    platform_leaf = evaluator.add_leaf(
        id="streaming_platform",
        desc="The streaming platform is specified (e.g., Netflix, Prime Video, Hulu, etc.)",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) confirm that the streaming platform for '{title}' is '{platform}'.",
        node=platform_leaf,
        sources=info_urls,
        additional_instruction="Confirm the named platform on which the title premiered."
    )

    personnel_node = evaluator.add_parallel(
        id="streaming_personnel",
        desc="Verify at least two personnel involved in the streaming item with their roles",
        parent=node,
        critical=True
    )

    persons = _two_persons(item.persons if item else [])
    for idx, p in enumerate(persons, start=1):
        evaluator.add_custom_node(
            result=bool(p.source_urls),
            id=f"streaming_person_{idx}_url",
            desc=f"URL reference confirming the {'first' if idx == 1 else 'second'} individual's involvement in the streaming item",
            parent=personnel_node,
            critical=True
        )

        leaf = evaluator.add_leaf(
            id=f"streaming_person_{idx}",
            desc=f"{'First' if idx == 1 else 'Second'} individual involved in the streaming item is identified with their specific role",
            parent=personnel_node,
            critical=True
        )
        all_sources = _merge_sources(p.source_urls, info_urls)
        await evaluator.verify(
            claim=f"The page(s) confirm that {p.name or ''} served as {p.role or ''} for the streaming title '{title}'.",
            node=leaf,
            sources=all_sources,
            additional_instruction="Confirm the person and role for this streaming title. Accept usual synonyms (e.g., creator/showrunner/writer)."
        )


async def verify_tv_episode(evaluator: Evaluator, parent, item: Optional[TVEpisodeItem]) -> None:
    node = evaluator.add_parallel(
        id="tv_episode_item",
        desc="Evaluate the TV series episode entry",
        parent=parent,
        critical=False
    )

    basic = evaluator.add_parallel(
        id="tv_episode_basic_info",
        desc="Verify basic identifying information for the TV episode",
        parent=node,
        critical=True
    )

    info_urls = item.info_urls if item else []
    series = (item.series_title or "") if item else ""
    episode = (item.episode_title or "") if item else ""
    air_date = (item.air_date or "") if item else ""
    network = (item.network or "") if item else ""

    evaluator.add_custom_node(
        result=bool(info_urls),
        id="tv_episode_basic_info_url",
        desc="URL reference supporting the basic information about the TV episode",
        parent=basic,
        critical=True
    )

    title_leaf = evaluator.add_leaf(
        id="tv_episode_title",
        desc="The series name and specific episode title are provided",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) confirm an episode titled '{episode}' from the TV series '{series}'.",
        node=title_leaf,
        sources=info_urls,
        additional_instruction="The evidence should clearly link the episode title to the named series."
    )

    date_leaf = evaluator.add_leaf(
        id="tv_episode_air_date",
        desc=f"The air date is provided and falls between {WINDOW_START}, and {WINDOW_END}",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) confirm that the episode '{episode}' of '{series}' aired on {air_date}, and this date falls between {WINDOW_START} and {WINDOW_END}.",
        node=date_leaf,
        sources=info_urls,
        additional_instruction="Confirm the linear TV air date (not streaming). Judge whether it lies within the window."
    )

    network_leaf = evaluator.add_leaf(
        id="tv_episode_network",
        desc="The broadcast or cable network is specified (e.g., CBS, NBC, ABC, HBO, etc.)",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) confirm that the network for this episode is '{network}'.",
        node=network_leaf,
        sources=info_urls,
        additional_instruction="Confirm the broadcast/cable network (e.g., ABC, NBC, FOX, CBS, HBO, FX)."
    )

    personnel_node = evaluator.add_parallel(
        id="tv_episode_personnel",
        desc="Verify at least two personnel involved in the TV episode with their roles",
        parent=node,
        critical=True
    )

    persons = _two_persons(item.persons if item else [])
    for idx, p in enumerate(persons, start=1):
        evaluator.add_custom_node(
            result=bool(p.source_urls),
            id=f"tv_episode_person_{idx}_url",
            desc=f"URL reference confirming the {'first' if idx == 1 else 'second'} individual's involvement in the TV episode",
            parent=personnel_node,
            critical=True
        )

        leaf = evaluator.add_leaf(
            id=f"tv_episode_person_{idx}",
            desc=f"{'First' if idx == 1 else 'Second'} individual involved in the TV episode is identified with their specific role",
            parent=personnel_node,
            critical=True
        )
        all_sources = _merge_sources(p.source_urls, info_urls)
        await evaluator.verify(
            claim=f"The page(s) confirm that {p.name or ''} served as {p.role or ''} for the episode '{episode}' of '{series}'.",
            node=leaf,
            sources=all_sources,
            additional_instruction="Confirm the person's role for this specific episode or series (e.g., guest star, writer, director)."
        )


async def verify_broadway_show(evaluator: Evaluator, parent, item: Optional[BroadwayShowItem]) -> None:
    node = evaluator.add_parallel(
        id="broadway_show_item",
        desc="Evaluate the Broadway show entry",
        parent=parent,
        critical=False
    )

    basic = evaluator.add_parallel(
        id="broadway_show_basic_info",
        desc="Verify basic identifying information for the Broadway show",
        parent=node,
        critical=True
    )

    title = (item.title or "") if item else ""
    venue = (item.venue or "") if item else ""
    participation_urls = item.participation_urls if item else []
    info_urls = item.info_urls if item else []

    # Participation URL presence (critical gating)
    evaluator.add_custom_node(
        result=bool(participation_urls),
        id="broadway_show_basic_info_url",
        desc="URL reference supporting the show's participation in NYC Broadway Week 2026",
        parent=basic,
        critical=True
    )

    title_leaf = evaluator.add_leaf(
        id="broadway_show_title",
        desc="The exact title of the Broadway show is provided",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) confirm the Broadway production's title is exactly '{title}'.",
        node=title_leaf,
        sources=_merge_sources(info_urls, participation_urls),
        additional_instruction="Confirm the production title as a Broadway show. Allow minor punctuation/case differences."
    )

    participation_leaf = evaluator.add_leaf(
        id="broadway_show_participation",
        desc=f"The show is verified as one of the 26 shows participating in NYC Broadway Week 2026 ({BROADWAY_WEEK_START} - {BROADWAY_WEEK_END})",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) confirm that the show '{title}' participated in NYC Broadway Week 2026, offering 2-for-1 tickets between {BROADWAY_WEEK_START} and {BROADWAY_WEEK_END}.",
        node=participation_leaf,
        sources=participation_urls,
        additional_instruction="The page should explicitly indicate participation in 'NYC Broadway Week 2026' (Jan 20–Feb 12, 2026) and that it was part of the official 2-for-1 ticket promotion. Prefer official listing pages (e.g., NYC Tourism/nycgo) or authoritative sources."
    )

    venue_leaf = evaluator.add_leaf(
        id="broadway_show_venue",
        desc="The Broadway theater venue or Broadway designation is specified",
        parent=basic,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) confirm that the Broadway venue/designation for '{title}' is '{venue}'.",
        node=venue_leaf,
        sources=_merge_sources(info_urls, participation_urls),
        additional_instruction="Confirm the named Broadway theater venue (e.g., Majestic Theatre) or explicit 'Broadway' designation for this production."
    )

    personnel_node = evaluator.add_parallel(
        id="broadway_show_personnel",
        desc="Verify at least two personnel currently involved in the Broadway show production with their roles",
        parent=node,
        critical=True
    )

    persons = _two_persons(item.persons if item else [])
    for idx, p in enumerate(persons, start=1):
        evaluator.add_custom_node(
            result=bool(p.source_urls),
            id=f"broadway_show_person_{idx}_url",
            desc=f"URL reference confirming the {'first' if idx == 1 else 'second'} individual's involvement in the Broadway show",
            parent=personnel_node,
            critical=True
        )

        leaf = evaluator.add_leaf(
            id=f"broadway_show_person_{idx}",
            desc=f"{'First' if idx == 1 else 'Second'} individual involved in the Broadway show is identified with their specific role",
            parent=personnel_node,
            critical=True
        )
        all_sources = _merge_sources(p.source_urls, info_urls, participation_urls)
        await evaluator.verify(
            claim=f"The page(s) confirm that {p.name or ''} served as {p.role or ''} in the Broadway production '{title}'.",
            node=leaf,
            sources=all_sources,
            additional_instruction="Confirm the person's role in this Broadway production (e.g., actor, director, choreographer, composer)."
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
    """
    Evaluate whether four entertainment items from different categories (theatrical film, streaming premiere,
    TV episode, Broadway show) released/occurred in the specified 2026 window are correctly identified
    with accurate titles, dates, platforms/venues/networks, and personnel with roles.
    """
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_entertainment_items(),
        template_class=EntertainmentExtraction,
        extraction_name="extracted_items"
    )

    # Optional context info
    evaluator.add_custom_info(
        info={
            "time_window": {"start": WINDOW_START, "end": WINDOW_END},
            "broadway_week_window": {"start": BROADWAY_WEEK_START, "end": BROADWAY_WEEK_END}
        },
        info_type="windows",
        info_name="evaluation_windows"
    )

    # Build and verify each category subtree
    await verify_theatrical_film(evaluator, root, extracted.theatrical_film)
    await verify_streaming_item(evaluator, root, extracted.streaming_item)
    await verify_tv_episode(evaluator, root, extracted.tv_episode)
    await verify_broadway_show(evaluator, root, extracted.broadway_show)

    return evaluator.get_summary()