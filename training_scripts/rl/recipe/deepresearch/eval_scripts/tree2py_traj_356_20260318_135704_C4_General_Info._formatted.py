import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thriller_2024_2025_film_verification"
TASK_DESCRIPTION = """
Identify a thriller film that was released between 2024 and 2025 and meets all of the following criteria:

1. The film's director must have previously directed a critically acclaimed film that received major award nominations (such as Academy Award, BAFTA, or equivalent)
2. The film must star at least two internationally recognized lead actors
3. At least one of the lead actors must be from Australia or the United Kingdom
4. The film must have premiered at a major international film festival (such as Toronto International Film Festival, Tribeca Film Festival, Cannes, Venice, or Sundance)
5. The film must have been released both in theaters and on a streaming platform
6. The film must be classified as a thriller
7. The theatrical release must have occurred in 2025
8. There must be a verifiable time gap of at least 300 days between the festival premiere and the theatrical release date

For this film, provide:
- The name of the film
- The director's name and one previously acclaimed film they directed
- The names of at least two lead actors and identify which one is from Australia/UK
- The film festival where it premiered and the premiere date
- The theatrical release date
- The streaming platform and streaming release date
- The exact number of days between the festival premiere and theatrical release
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FilmExtraction(BaseModel):
    # Core film identification
    film_name: Optional[str] = None
    film_urls: List[str] = Field(default_factory=list)

    # Director and prior acclaimed film
    director_name: Optional[str] = None
    prior_film_title: Optional[str] = None
    prior_film_urls: List[str] = Field(default_factory=list)  # general prior film page(s)
    prior_film_awards_urls: List[str] = Field(default_factory=list)  # evidence of major award nominations
    prior_film_critical_acclaim_urls: List[str] = Field(default_factory=list)  # evidence of "critical acclaim"

    # Lead actors (at least two)
    lead_actor_1_name: Optional[str] = None
    lead_actor_1_urls: List[str] = Field(default_factory=list)
    lead_actor_2_name: Optional[str] = None
    lead_actor_2_urls: List[str] = Field(default_factory=list)

    # Australia/UK condition
    aus_uk_actor_name: Optional[str] = None  # which lead actor is from AU/UK (must be one of the above)
    aus_uk_actor_urls: List[str] = Field(default_factory=list)

    # Festival premiere
    festival_name: Optional[str] = None
    festival_premiere_date: Optional[str] = None  # as written in the answer
    festival_premiere_urls: List[str] = Field(default_factory=list)  # a page showing the film premiered at that festival
    festival_major_urls: List[str] = Field(default_factory=list)  # evidence that the festival is a major international one

    # Releases
    theatrical_release_date: Optional[str] = None  # must be in 2025
    theatrical_release_urls: List[str] = Field(default_factory=list)
    streaming_platform: Optional[str] = None
    streaming_release_date: Optional[str] = None
    streaming_urls: List[str] = Field(default_factory=list)

    # Genre
    genre_label: Optional[str] = None
    genre_urls: List[str] = Field(default_factory=list)

    # Reported gap days (exact number claimed in the answer)
    gap_days_reported: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_film_data() -> str:
    return """
Extract structured information for exactly one (1) film described in the answer that aims to satisfy the task requirements.

Return a JSON object with these fields (use null for any missing string field; use [] for any missing list field). Extract ONLY from the answer text; do not invent.

Required fields:
- film_name: the film's title
- film_urls: list of URLs in the answer that directly describe the film (e.g., official site, distributor/press page, Wikipedia, IMDb)

- director_name: the film’s director
- prior_film_title: one previously directed film by this director (claimed to be acclaimed and major-award-nominated)
- prior_film_urls: URLs about the prior film (e.g., Wikipedia/IMDb/official page)
- prior_film_awards_urls: URLs supporting that the prior film received MAJOR award nominations (Academy Awards/Oscars, BAFTA, Golden Globes, Cannes/Venice awards, or comparable)
- prior_film_critical_acclaim_urls: URLs supporting that the prior film is “critically acclaimed” (e.g., reliable press coverage, Wikipedia statements, high aggregator pages)

- lead_actor_1_name: first lead actor named
- lead_actor_1_urls: URLs supporting lead actor’s profile/filmography (Wikipedia or IMDb strongly preferred)
- lead_actor_2_name: second lead actor named
- lead_actor_2_urls: URLs supporting lead actor’s profile/filmography

- aus_uk_actor_name: which named lead actor (from lead_actor_1_name or lead_actor_2_name) is from Australia or the United Kingdom (identify exactly one if present)
- aus_uk_actor_urls: URLs supporting the actor’s nationality (Wikipedia/official bio preferred)

- festival_name: the major international film festival where the film premiered
- festival_premiere_date: the festival premiere date as written in the answer (e.g., “September 8, 2024”)
- festival_premiere_urls: URLs that show this film premiered at that festival (festival program page, press release, or reputable coverage)
- festival_major_urls: URLs that establish the festival is a major international film festival (Wikipedia or official site acceptable)

- theatrical_release_date: theatrical release date (the answer should claim it occurred in 2025)
- theatrical_release_urls: URLs supporting the theatrical release date (distributor/press, Wikipedia, BoxOfficeMojo, etc.)

- streaming_platform: name of the streaming platform (e.g., Netflix, Amazon Prime Video, Hulu, Max, Apple TV+, Disney+)
- streaming_release_date: date when it was released/available on streaming
- streaming_urls: URLs supporting the streaming platform and date (platform page or reputable coverage)

- genre_label: the genre label as given in the answer (should indicate “thriller”, e.g., “thriller” / “psychological thriller” / “crime thriller”)
- genre_urls: URLs supporting that the film is classified as a thriller (Wikipedia/IMDb or official sources)

- gap_days_reported: the exact number of days between the festival premiere and theatrical release as reported in the answer (numeric string preferred; if embedded in text, extract the digits)

Rules:
- Include only URLs explicitly present in the answer. If none are provided for a field, return an empty list.
- Preserve date strings exactly as in the answer (do not reformat).
- If more than two lead actors are listed, extract the first two as lead_actor_1 and lead_actor_2.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _first_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(lst or [])
    return _dedup(merged)


_DATE_FORMATS = [
    "%B %d, %Y",      # January 31, 2025
    "%b %d, %Y",      # Jan 31, 2025
    "%Y-%m-%d",       # 2025-01-31
    "%d %B %Y",       # 31 January 2025
    "%d %b %Y",       # 31 Jan 2025
    "%B %Y",          # January 2025
    "%b %Y",          # Jan 2025
    "%m/%d/%Y",       # 01/31/2025
    "%d/%m/%Y",       # 31/01/2025
]


def _parse_date_safe(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    # Remove ordinal suffixes like "September 8th, 2024"
    s = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', s)
    for fmt in _DATE_FORMATS:
        try:
            # If month-year format (%B %Y or %b %Y), assume day=1 for diff purposes
            if fmt in ("%B %Y", "%b %Y"):
                dt = datetime.strptime(s, fmt)
                return datetime(dt.year, dt.month, 1)
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # Fallback: try to extract YYYY-MM-DD-like pattern
    m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', s)
    if m:
        try:
            y, mo, d = map(int, m.groups())
            return datetime(y, mo, d)
        except Exception:
            pass
    # Try Month Day, Year pattern approx
    m2 = re.search(r'([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})', s)
    if m2:
        try:
            month_str, day_str, year_str = m2.groups()
            day = int(day_str)
            year = int(year_str)
            for fmt in ("%B", "%b"):
                try:
                    mo = datetime.strptime(month_str, fmt).month
                    return datetime(year, mo, day)
                except Exception:
                    continue
        except Exception:
            pass
    return None


def _extract_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r'-?\d+', s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, info: FilmExtraction) -> None:
    # Create top-level critical node under root to mirror rubric
    film_info_node = evaluator.add_parallel(
        id="Film_Information",
        desc="Verify the identified film satisfies all specified constraints and that all requested fields are provided.",
        parent=evaluator.root,
        critical=True,
    )

    # ---------------- Film_Name (existence) ----------------
    evaluator.add_custom_node(
        result=bool(info.film_name and info.film_name.strip()),
        id="Film_Name",
        desc="Provide the name of the film.",
        parent=film_info_node,
        critical=True,
    )

    # ---------------- Director_and_Prior_Film --------------
    dir_prior_node = evaluator.add_parallel(
        id="Director_and_Prior_Film",
        desc="Provide the director’s name and one previously directed film, and verify that prior film is critically acclaimed and received major award nominations.",
        parent=film_info_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.director_name and info.director_name.strip()),
        id="Director_Name",
        desc="Director name is provided.",
        parent=dir_prior_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.prior_film_title and info.prior_film_title.strip()),
        id="Prior_Film_Title",
        desc="One prior film directed by the director is named.",
        parent=dir_prior_node,
        critical=True,
    )

    # Major award nominations for prior film (URL-backed)
    prior_award_leaf = evaluator.add_leaf(
        id="Prior_Film_Major_Award_Nominations",
        desc="The named prior film received major award nominations (e.g., Academy Awards, BAFTA, or equivalent major awards).",
        parent=dir_prior_node,
        critical=True,
    )
    prior_award_claim = (
        f"The film '{info.prior_film_title}' received nominations for at least one major award such as "
        f"the Oscars (Academy Awards), BAFTA, Golden Globes, Cannes/Venice awards, or an equivalent major international award."
    )
    await evaluator.verify(
        claim=prior_award_claim,
        node=prior_award_leaf,
        sources=_first_sources(info.prior_film_awards_urls, info.prior_film_urls),
        additional_instruction="Confirm explicit mention of nominations (not just wins) for recognized major awards. Wikipedia award sections and reputable press are acceptable.",
    )

    # Critical acclaim for prior film (URL-backed)
    prior_acclaim_leaf = evaluator.add_leaf(
        id="Prior_Film_Critical_Acclaim",
        desc="The named prior film is critically acclaimed (demonstrable via reputable critical reception/recognition).",
        parent=dir_prior_node,
        critical=True,
    )
    prior_acclaim_claim = (
        f"The film '{info.prior_film_title}' is widely described as 'critically acclaimed' or clearly recognized by reputable critics/publications or high aggregator scores."
    )
    await evaluator.verify(
        claim=prior_acclaim_claim,
        node=prior_acclaim_leaf,
        sources=_first_sources(info.prior_film_critical_acclaim_urls, info.prior_film_urls),
        additional_instruction="Accept evidence such as reputable outlets using the phrase 'critical acclaim' or high-profile recognitions; aggregator pages (e.g., Metacritic, Rotten Tomatoes) indicating strong critical reception also acceptable.",
    )

    # ---------------- Lead_Actors --------------------------
    lead_node = evaluator.add_parallel(
        id="Lead_Actors",
        desc="Provide at least two lead actors and verify international recognition plus Australia/UK criterion.",
        parent=film_info_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.lead_actor_1_name and info.lead_actor_1_name.strip()),
        id="Lead_Actor_1_Name",
        desc="Name of the first lead actor is provided.",
        parent=lead_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.lead_actor_2_name and info.lead_actor_2_name.strip()),
        id="Lead_Actor_2_Name",
        desc="Name of the second lead actor is provided.",
        parent=lead_node,
        critical=True,
    )

    # International recognition checks (URL-backed)
    la1_rec_leaf = evaluator.add_leaf(
        id="Lead_Actor_1_International_Recognition",
        desc="The first named lead actor is internationally recognized.",
        parent=lead_node,
        critical=True,
    )
    la1_rec_claim = (
        f"{info.lead_actor_1_name} is an internationally recognized actor (e.g., widely known with international credits, "
        f"awards/major nominations, or prominent roles in internationally released films/series)."
    )
    await evaluator.verify(
        claim=la1_rec_claim,
        node=la1_rec_leaf,
        sources=_first_sources(info.lead_actor_1_urls),
        additional_instruction="Evidence like a comprehensive Wikipedia page, major award nominations/wins, or starring roles in internationally released films is sufficient.",
    )

    la2_rec_leaf = evaluator.add_leaf(
        id="Lead_Actor_2_International_Recognition",
        desc="The second named lead actor is internationally recognized.",
        parent=lead_node,
        critical=True,
    )
    la2_rec_claim = (
        f"{info.lead_actor_2_name} is an internationally recognized actor (e.g., widely known with international credits, "
        f"awards/major nominations, or prominent roles in internationally released films/series)."
    )
    await evaluator.verify(
        claim=la2_rec_claim,
        node=la2_rec_leaf,
        sources=_first_sources(info.lead_actor_2_urls),
        additional_instruction="Evidence like a comprehensive Wikipedia page, major award nominations/wins, or starring roles in internationally released films is sufficient.",
    )

    # Australia/UK actor check (URL-backed). Also soft-check that the named actor is among the leads.
    au_uk_leaf = evaluator.add_leaf(
        id="Australia_UK_Lead_Actor_Check",
        desc="At least one of the named lead actors is from Australia or the United Kingdom, and the answer identifies which actor satisfies this.",
        parent=lead_node,
        critical=True,
    )
    au_uk_claim = (
        f"At least one of the lead actors is Australian or British; specifically, {info.aus_uk_actor_name} is from Australia or the United Kingdom."
    )
    await evaluator.verify(
        claim=au_uk_claim,
        node=au_uk_leaf,
        sources=_first_sources(info.aus_uk_actor_urls, info.lead_actor_1_urls, info.lead_actor_2_urls),
        additional_instruction=(
            "Confirm nationality from reliable sources (Wikipedia/official bios). "
            "For the purpose of this check, allow British, English, Scottish, Welsh, or Northern Irish as UK."
        ),
    )

    # ---------------- Festival_Premiere --------------------
    fest_node = evaluator.add_parallel(
        id="Festival_Premiere",
        desc="Provide the major international festival premiere and the premiere date.",
        parent=film_info_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.festival_name and info.festival_name.strip()),
        id="Festival_Name",
        desc="Festival where the film premiered is provided.",
        parent=fest_node,
        critical=True,
    )

    fest_major_leaf = evaluator.add_leaf(
        id="Festival_Is_Major_International",
        desc="The named festival qualifies as a major international film festival (e.g., Toronto, Tribeca, Cannes, Venice, Sundance, or comparable).",
        parent=fest_node,
        critical=True,
    )
    fest_major_claim = (
        f"{info.festival_name} is a major international film festival comparable in stature to Toronto (TIFF), Cannes, Venice, Sundance, Tribeca, or Berlin."
    )
    await evaluator.verify(
        claim=fest_major_claim,
        node=fest_major_leaf,
        sources=_first_sources(info.festival_major_urls, info.festival_premiere_urls),
        additional_instruction="Use reputable descriptions (festival's Wikipedia/official site or reputable media) to confirm the festival's international significance.",
    )

    evaluator.add_custom_node(
        result=bool(info.festival_premiere_date and info.festival_premiere_date.strip()),
        id="Premiere_Date",
        desc="Festival premiere date is provided.",
        parent=fest_node,
        critical=True,
    )

    # ---------------- Release_Modes ------------------------
    release_node = evaluator.add_parallel(
        id="Release_Modes",
        desc="Verify theatrical and streaming releases and provide required dates/platform.",
        parent=film_info_node,
        critical=True,
    )

    theatrical_leaf = evaluator.add_leaf(
        id="Theatrical_Release_Date_2025",
        desc="The theatrical release date is provided and occurs in 2025.",
        parent=release_node,
        critical=True,
    )
    theatrical_claim = (
        f"The film had a theatrical release on {info.theatrical_release_date}, and that date is in 2025."
    )
    await evaluator.verify(
        claim=theatrical_claim,
        node=theatrical_leaf,
        sources=_first_sources(info.theatrical_release_urls, info.film_urls),
        additional_instruction="Confirm the theatrical release took place and that the year is 2025.",
    )

    streaming_leaf = evaluator.add_leaf(
        id="Streaming_Platform_and_Date",
        desc="A streaming platform is named and a streaming release date is provided.",
        parent=release_node,
        critical=True,
    )
    streaming_claim = (
        f"The film was released on streaming platform {info.streaming_platform} on {info.streaming_release_date}."
    )
    await evaluator.verify(
        claim=streaming_claim,
        node=streaming_leaf,
        sources=_first_sources(info.streaming_urls),
        additional_instruction="Verify both the platform name and the streaming availability/release date from a credible source (platform page or reputable media).",
    )

    # ---------------- Genre_Classification -----------------
    genre_leaf = evaluator.add_leaf(
        id="Genre_Classification",
        desc="The film is classified as a thriller.",
        parent=film_info_node,
        critical=True,
    )
    genre_claim = (
        f"The film '{info.film_name}' is classified as a thriller (including subtypes like psychological thriller, crime thriller, techno-thriller, etc.)."
    )
    await evaluator.verify(
        claim=genre_claim,
        node=genre_leaf,
        sources=_first_sources(info.genre_urls, info.film_urls),
        additional_instruction="Accept if reputable sources (Wikipedia/IMDb/official) classify it as 'thriller' or a clear thriller subgenre.",
    )

    # ---------------- Release_Timeline_Gap -----------------
    # Break into concrete leaves: (1) exact days number correctness (2) >= 300 days
    gap_node = evaluator.add_sequential(
        id="Release_Timeline_Gap",
        desc="Provide the exact number of days between the festival premiere date and the theatrical release date, and verify the gap is at least 300 days.",
        parent=film_info_node,
        critical=True,
    )

    # Compute difference (if possible)
    dt_premiere = _parse_date_safe(info.festival_premiere_date)
    dt_theatrical = _parse_date_safe(info.theatrical_release_date)
    computed_gap: Optional[int] = None
    if dt_premiere and dt_theatrical:
        computed_gap = (dt_theatrical.date() - dt_premiere.date()).days

    reported_gap = _extract_int(info.gap_days_reported)
    gap_number_correct = (reported_gap is not None and computed_gap is not None and reported_gap == computed_gap)

    evaluator.add_custom_node(
        result=bool(gap_number_correct),
        id="Gap_Days_Exact_Number_Correct",
        desc="The reported exact number of days between the festival premiere and theatrical release matches the computed difference.",
        parent=gap_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(computed_gap is not None and computed_gap >= 300),
        id="Gap_At_Least_300_Days",
        desc="The gap between the festival premiere date and the theatrical release date is at least 300 days.",
        parent=gap_node,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured film info from the answer
    extracted: FilmExtraction = await evaluator.extract(
        prompt=prompt_extract_film_data(),
        template_class=FilmExtraction,
        extraction_name="film_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()