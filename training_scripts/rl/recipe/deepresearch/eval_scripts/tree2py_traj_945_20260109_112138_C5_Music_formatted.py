import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "billboard_hot100_2024_number_one_artists"
TASK_DESCRIPTION = (
    "Identify three different artists who each had a song reach #1 on the Billboard Hot 100 chart during 2024. "
    "For each artist, provide: (1) The artist's name, (2) The exact title of the song that reached #1, "
    "(3) The date or month when the song first reached #1 on the Billboard Hot 100, "
    "(4) The total number of weeks the song spent at #1, and "
    "(5) One notable achievement or record associated with that song's chart performance. "
    "The three artists must be distinct individuals or groups, and all information must be verifiable through "
    "Billboard Hot 100 chart data. Include reference URLs that support your findings for each artist."
)


# ----------------------------- Data Models --------------------------------- #
class ArtistEntry(BaseModel):
    artist_name: Optional[str] = None
    song_title: Optional[str] = None
    first_number_one_date_or_month: Optional[str] = None
    weeks_at_number_one: Optional[str] = None
    notable_achievement: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ArtistsExtraction(BaseModel):
    artists: List[ArtistEntry] = Field(default_factory=list)


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_artists() -> str:
    return (
        "Extract all artist-song entries mentioned in the answer that claim a Billboard Hot 100 No. 1 in 2024. "
        "For each entry, extract the following fields exactly as stated in the answer:\n"
        "1) artist_name: The artist or group name.\n"
        "2) song_title: The exact song title that reached #1 on the Billboard Hot 100.\n"
        "3) first_number_one_date_or_month: The date (e.g., 'January 20, 2024') or month (e.g., 'January 2024') "
        "when the song first reached #1 on the Billboard Hot 100.\n"
        "4) weeks_at_number_one: The total number of weeks the song spent at #1 (keep as a string, e.g., '7', '7 weeks').\n"
        "5) notable_achievement: One notable achievement/record associated with the song’s chart performance.\n"
        "6) reference_urls: All URLs explicitly provided in the answer that support this entry. Extract only URLs; "
        "accept markdown links and plain URLs; ensure full URLs with protocol. If optional contextual sources (e.g., "
        "Billboard news posts, chart pages) are mentioned with domain-only references, extract only if a concrete URL "
        "is provided in the answer.\n\n"
        "Return a JSON object with an array 'artists'. Each element must include the keys above. "
        "If any field is missing for an entry, set it to null (or empty list for reference_urls). "
        "Do not invent or infer any data not in the answer."
    )


# ------------------------------ Helpers ------------------------------------ #
def is_billboard_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        return ("billboard.com" in host) or ("charts.billboard.com" in host)
    except Exception:
        return False


def normalize_urls(urls: List[str]) -> List[str]:
    cleaned = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        parsed = urlparse(u)
        if not parsed.scheme:
            u = "http://" + u
        cleaned.append(u)
    return cleaned


# --------------------------- Verification Logic ---------------------------- #
async def verify_artist_entry(
    evaluator: Evaluator,
    parent_node,
    entry: ArtistEntry,
    idx: int,
) -> None:
    """
    Build verification subtree for a single artist entry.
    """
    artist_num = idx + 1
    artist_node = evaluator.add_parallel(
        id=f"Artist_{artist_num}",
        desc=f"Artist #{artist_num} entry verification",
        parent=parent_node,
        critical=False,
    )

    # Normalize URLs once
    sources_all = normalize_urls(entry.reference_urls or [])
    billboard_urls = [u for u in sources_all if is_billboard_url(u)]

    # References group (critical)
    refs_node = evaluator.add_parallel(
        id=f"Artist_{artist_num}_References",
        desc=f"Reference URLs sufficient to support the claims for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )

    has_urls_node = evaluator.add_custom_node(
        result=(len(sources_all) > 0),
        id=f"Artist_{artist_num}_Has_Reference_URLs",
        desc=f"Includes at least one reference URL for artist #{artist_num}",
        parent=refs_node,
        critical=True,
    )

    has_billboard_url_node = evaluator.add_custom_node(
        result=(len(billboard_urls) > 0),
        id=f"Artist_{artist_num}_Has_Billboard_URL",
        desc=f"At least one provided URL is a Billboard domain for artist #{artist_num}",
        parent=refs_node,
        critical=True,
    )

    # Chart claims support sub-group under references (critical)
    chart_support_node = evaluator.add_parallel(
        id=f"Artist_{artist_num}_URLs_Support_Chart_Claims",
        desc=f"Provided URLs support song title, first #1 timing, and weeks at #1 for artist #{artist_num}",
        parent=refs_node,
        critical=True,
    )

    # 1) Artist name provided (critical existence)
    evaluator.add_custom_node(
        result=(entry.artist_name is not None and entry.artist_name.strip() != ""),
        id=f"Artist_{artist_num}_Name",
        desc=f"Artist name is provided for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )

    # 2) Song title exact (critical; verify via URLs)
    song_title_leaf = evaluator.add_leaf(
        id=f"Artist_{artist_num}_Song_Title_Exact",
        desc=f"Exact song title is correct per Billboard Hot 100 for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Billboard Hot 100 lists the song title exactly as '{entry.song_title}'.",
        node=song_title_leaf,
        sources=sources_all,
        additional_instruction=(
            "Check the official Billboard Hot 100 listing(s) or Billboard articles. "
            "Treat the title as exact even if minor punctuation or capitalization differs. "
            "Reject if the page indicates a different song or artist."
        ),
        extra_prerequisites=[has_urls_node],
    )

    # Duplicate support for title inside references/chart-support to satisfy rubric structure
    urls_support_title_leaf = evaluator.add_leaf(
        id=f"Artist_{artist_num}_URLs_Support_Song_Title",
        desc=f"Provided URL(s) support the exact song title for artist #{artist_num}",
        parent=chart_support_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided sources explicitly show the song title '{entry.song_title}' as the Hot 100 entry that reached #1.",
        node=urls_support_title_leaf,
        sources=sources_all,
        additional_instruction=(
            "Confirm the song title on the referenced Billboard chart or article. "
            "Minor formatting differences (quotes, case) are acceptable."
        ),
        extra_prerequisites=[has_urls_node],
    )

    # 3) First reach #1 date or month (critical; existence + verification)
    evaluator.add_custom_node(
        result=(entry.first_number_one_date_or_month is not None and entry.first_number_one_date_or_month.strip() != ""),
        id=f"Artist_{artist_num}_First_Reach_Number_One_DateOrMonth_Provided",
        desc=f"Provides the date or month for first reach #1 for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )
    first_date_leaf = evaluator.add_leaf(
        id=f"Artist_{artist_num}_First_Reach_Number_One_DateOrMonth_Accurate",
        desc=f"The first #1 timing (date or month) is accurate per Billboard for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The song first reached #1 on the Billboard Hot 100 in {entry.first_number_one_date_or_month}.",
        node=first_date_leaf,
        sources=sources_all,
        additional_instruction=(
            "Validate the first date/month the track hit No. 1. Accept a precise week-of date or month-year. "
            "Use Billboard chart history or official Billboard reporting."
        ),
        extra_prerequisites=[has_urls_node],
    )

    urls_support_first_date_leaf = evaluator.add_leaf(
        id=f"Artist_{artist_num}_URLs_Support_First_Reach_Timing",
        desc=f"Provided URLs support the first #1 timing (date/month) for artist #{artist_num}",
        parent=chart_support_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The sources confirm the song first reached #1 in {entry.first_number_one_date_or_month}.",
        node=urls_support_first_date_leaf,
        sources=sources_all,
        additional_instruction=(
            "Cross-check Billboard chart week listings or articles to confirm the first No. 1 date/month."
        ),
        extra_prerequisites=[has_urls_node],
    )

    # 4) Explicit check: number-one occurs in 2024 (critical)
    occurs_2024_leaf = evaluator.add_leaf(
        id=f"Artist_{artist_num}_Number_One_Occurs_In_2024",
        desc=f"The song reached #1 during the 2024 calendar year for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This song reached #1 on the Billboard Hot 100 during the 2024 calendar year.",
        node=occurs_2024_leaf,
        sources=sources_all,
        additional_instruction=(
            "Confirm that at least one No. 1 week for this song falls in 2024. "
            "Billboard chart archives or Billboard news posts announcing the No. 1 are acceptable."
        ),
        extra_prerequisites=[has_urls_node],
    )

    # 5) Weeks at #1 (critical; existence + verification)
    evaluator.add_custom_node(
        result=(entry.weeks_at_number_one is not None and entry.weeks_at_number_one.strip() != ""),
        id=f"Artist_{artist_num}_Weeks_At_Number_One_Provided",
        desc=f"Provides the total weeks at #1 on Hot 100 for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )
    weeks_leaf = evaluator.add_leaf(
        id=f"Artist_{artist_num}_Weeks_At_Number_One_Accurate",
        desc=f"Weeks at #1 is accurate per Billboard records for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The song spent {entry.weeks_at_number_one} weeks at #1 on the Billboard Hot 100.",
        node=weeks_leaf,
        sources=sources_all,
        additional_instruction=(
            "Check Billboard chart history or cumulative stats. "
            "Allow numeric equivalence (e.g., '7' vs 'seven')."
        ),
        extra_prerequisites=[has_urls_node],
    )

    urls_support_weeks_leaf = evaluator.add_leaf(
        id=f"Artist_{artist_num}_URLs_Support_Weeks_At_Number_One",
        desc=f"Provided URLs support the total weeks at #1 for artist #{artist_num}",
        parent=chart_support_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The sources explicitly support that the song spent {entry.weeks_at_number_one} weeks at #1.",
        node=urls_support_weeks_leaf,
        sources=sources_all,
        additional_instruction=(
            "Confirm the total number of weeks at No. 1 from Billboard charts/coverage."
        ),
        extra_prerequisites=[has_urls_node],
    )

    # 6) Notable achievement (critical; existence + verification via Billboard)
    evaluator.add_custom_node(
        result=(entry.notable_achievement is not None and entry.notable_achievement.strip() != ""),
        id=f"Artist_{artist_num}_Notable_Achievement_Provided",
        desc=f"Provides a notable achievement/record for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )
    notable_leaf_general = evaluator.add_leaf(
        id=f"Artist_{artist_num}_Notable_Achievement_Accurate",
        desc=f"Notable achievement/record is factually correct for artist #{artist_num}",
        parent=artist_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Notable achievement: {entry.notable_achievement}",
        node=notable_leaf_general,
        sources=sources_all,
        additional_instruction=(
            "Confirm the stated achievement is accurate regarding the song’s Billboard Hot 100 performance."
        ),
        extra_prerequisites=[has_urls_node],
    )

    notable_billboard_leaf = evaluator.add_leaf(
        id=f"Artist_{artist_num}_URLs_Support_Notable_Achievement_Via_Billboard",
        desc=f"At least one Billboard URL supports the notable achievement for artist #{artist_num}",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Billboard confirms: {entry.notable_achievement}",
        node=notable_billboard_leaf,
        sources=billboard_urls,
        additional_instruction=(
            "Only use Billboard-owned domains (billboard.com or charts.billboard.com). "
            "Verify the achievement as stated."
        ),
        extra_prerequisites=[has_billboard_url_node],
    )


# ----------------------------- Main Function -------------------------------- #
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
        prompt=prompt_extract_artists(),
        template_class=ArtistsExtraction,
        extraction_name="artists_extraction",
    )

    # Record custom info about extraction size
    evaluator.add_custom_info(
        {"extracted_entries_count": len(extracted.artists)},
        info_type="stats",
        info_name="extraction_stats",
    )

    # Global requirements (critical)
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Global requirements applying to the whole response",
        parent=root,
        critical=True,
    )

    # Exactly three artist entries
    exactly_three = evaluator.add_custom_node(
        result=(len(extracted.artists) == 3),
        id="Exactly_Three_Artist_Entries",
        desc="Exactly three artist entries are provided.",
        parent=global_node,
        critical=True,
    )

    # Artists are distinct (check among the first three if available; fail otherwise)
    names_for_distinct = [a.artist_name.strip().lower() for a in extracted.artists[:3] if a.artist_name]
    artists_distinct = (len(names_for_distinct) == 3) and (len(set(names_for_distinct)) == 3)
    evaluator.add_custom_node(
        result=artists_distinct,
        id="Artists_Are_Distinct",
        desc="All three artists are distinct (no repeated artist).",
        parent=global_node,
        critical=True,
    )

    # Prepare entries to evaluate: first three, padded with empty if fewer
    selected: List[ArtistEntry] = list(extracted.artists[:3])
    while len(selected) < 3:
        selected.append(ArtistEntry())

    # Build per-artist verification subtrees (non-critical at root for partial credit)
    for i, entry in enumerate(selected):
        await verify_artist_entry(evaluator, root, entry, i)

    return evaluator.get_summary()