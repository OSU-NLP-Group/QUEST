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
TASK_ID = "mama_2025_song_of_the_year"
TASK_DESCRIPTION = """
Identify the Song of the Year (VISA Song of the Year) winner at the 2025 MAMA Awards. Provide the song title, the artist(s), the location where the awards ceremony was held, and the dates of the event.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MAMA2025Extraction(BaseModel):
    """
    Structured extraction of the answer's key fields for 2025 MAMA Awards (VISA Song of the Year).
    """
    song_title: Optional[str] = None
    artists: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    location_venue: Optional[str] = None
    location_city: Optional[str] = None
    location_country: Optional[str] = None
    dates: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_mama_2025() -> str:
    return """
    You must extract the specific information the answer gives about the 2025 MAMA Awards (Mnet Asian Music Awards) 'VISA Song of the Year'.
    Extract exactly what is stated in the answer text; do not infer or invent.

    Required fields:
    - song_title: The exact title of the winning song the answer claims won the 2025 MAMA 'VISA Song of the Year'. If missing, return null.
    - artists: A list of the artist name(s) as stated in the answer (e.g., soloist, group name, and/or featured artists if explicitly provided). If missing, return an empty list.
    - location: A concise free-text location description if the answer provides one string (e.g., "Tokyo Dome, Tokyo, Japan"). If not provided as a single description, return null.
    - location_venue: Venue name, if provided (e.g., "Tokyo Dome"). If not provided, return null.
    - location_city: City (and state/prefecture if given) where the ceremony took place, if provided. If not provided, return null.
    - location_country: Country where the ceremony took place, if provided. If not provided, return null.
    - dates: The date(s) of the 2025 MAMA Awards event as stated in the answer (e.g., "November 25–26, 2025"). If not provided, return null.
    - sources: An array of all citation URLs, such as official MAMA/Mnet pages or reputable music media (e.g., Billboard, Variety, NME, Rolling Stone, Soompi). Extract only URLs explicitly present in the answer text (plain URLs or links in markdown). If none, return an empty array.

    Do not normalize names or dates; capture exactly as the answer wrote them.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _join_nonempty(parts: List[Optional[str]], sep: str = ", ") -> Optional[str]:
    filtered = [p.strip() for p in parts if p and str(p).strip()]
    return sep.join(filtered) if filtered else None


def _artists_to_str(artists: List[str]) -> str:
    if not artists:
        return ""
    # Keep order as provided by the answer
    return ", ".join([a.strip() for a in artists if a and a.strip()])


def _compose_location_text(extracted: MAMA2025Extraction) -> str:
    if extracted.location and extracted.location.strip():
        return extracted.location.strip()
    return _join_nonempty([extracted.location_venue, extracted.location_city, extracted.location_country]) or ""


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
    Evaluate an answer for the 2025 MAMA Awards 'VISA Song of the Year' task.
    The evaluation checks:
    - Winning song title provided and correct (per cited sources).
    - Winning artist(s) provided and correct (per cited sources).
    - Event location provided and correct (per cited sources).
    - Event dates provided and correct (per cited sources).
    - Verifiable sourcing: at least one official MAMA/Mnet page or reputable music publication among the provided URLs
      that reports the specified winner and event details.
    """
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

    # Extraction phase
    extracted: MAMA2025Extraction = await evaluator.extract(
        prompt=prompt_extract_mama_2025(),
        template_class=MAMA2025Extraction,
        extraction_name="mama_2025_extraction",
    )

    # Prepare commonly used strings
    artists_str = _artists_to_str(extracted.artists)
    location_text = _compose_location_text(extracted)
    dates_text = (extracted.dates or "").strip()
    source_urls = extracted.sources or []

    # Build the verification tree according to rubric
    # Top-level critical node aggregating all required checks
    top_node = evaluator.add_parallel(
        id="2025_MAMA_Awards_VISA_Song_of_the_Year_Response",
        desc="Evaluate whether the answer correctly identifies the 2025 MAMA Awards (Mnet Asian Music Awards) VISA Song of the Year winner and provides accurate event details with verifiable sourcing.",
        parent=root,
        critical=True,
    )

    # Winner information (song title + artists)
    winner_info = evaluator.add_parallel(
        id="Winner_Information",
        desc="Answer provides the required winner details for the 2025 MAMA Awards 'Song of the Year (VISA Song of the Year)'.",
        parent=top_node,
        critical=True,
    )

    # Winner song title provided and correct
    winner_song_title_leaf = evaluator.add_leaf(
        id="Winner_Song_Title_Provided_And_Correct",
        desc="Provides the winning song title and it is factually correct for the 2025 VISA Song of the Year award.",
        parent=winner_info,
        critical=True,
    )
    song_title_claim = (
        f"The 2025 MAMA Awards (Mnet Asian Music Awards) 'VISA Song of the Year' winning song title is '{(extracted.song_title or '').strip()}'."
    )
    await evaluator.verify(
        claim=song_title_claim,
        node=winner_song_title_leaf,
        sources=source_urls if source_urls else None,
        additional_instruction=(
            "Verify this ONLY using the provided URLs (if any). If no URLs are provided, you must judge the claim as not supported/incorrect. "
            "Look for explicit mentions of the 2025 MAMA Awards (Mnet Asian Music Awards) 'Song of the Year' winner. "
            "Treat minor punctuation and case differences as equivalent; however, the song title must clearly match. "
            "If the extracted song title is empty or null, judge as incorrect."
        ),
    )

    # Winner artists provided and correct
    winner_artists_leaf = evaluator.add_leaf(
        id="Winner_Artists_Provided_And_Correct",
        desc="Provides the winning artist(s) and they are factually correct for the 2025 VISA Song of the Year award.",
        parent=winner_info,
        critical=True,
    )
    artists_claim = (
        f"The 2025 MAMA Awards 'VISA Song of the Year' winning artist(s) are exactly: {artists_str if artists_str else '∅'}."
    )
    await evaluator.verify(
        claim=artists_claim,
        node=winner_artists_leaf,
        sources=source_urls if source_urls else None,
        additional_instruction=(
            "Verify this ONLY using the provided URLs (if any). If no URLs are provided, you must judge the claim as not supported/incorrect. "
            "Confirm that the listed artist(s) match the credited performer(s) for the 2025 MAMA 'Song of the Year' winner. "
            "Allow minor formatting variations (e.g., '&' vs 'and', presence of 'feat.' or parentheses), but the set of credited artists must be equivalent. "
            "If the extracted artist list is empty, judge as incorrect."
        ),
    )

    # Event location provided and correct
    event_location_leaf = evaluator.add_leaf(
        id="Event_Location_Provided_And_Correct",
        desc="States the location/venue and host city (or equivalent location description) for where the 2025 MAMA Awards ceremony was held, and it is factually correct.",
        parent=top_node,
        critical=True,
    )
    location_claim = (
        f"The 2025 MAMA Awards ceremony was held at/in '{location_text if location_text else '∅'}'."
    )
    await evaluator.verify(
        claim=location_claim,
        node=event_location_leaf,
        sources=source_urls if source_urls else None,
        additional_instruction=(
            "Verify this ONLY using the provided URLs (if any). If no URLs are provided, you must judge the claim as not supported/incorrect. "
            "Accept reasonable equivalences between venue/city/country naming (e.g., abbreviations, alternate romanizations). "
            "Ensure the page refers specifically to the 2025 MAMA Awards ceremony location. "
            "If the extracted location text is empty, judge as incorrect."
        ),
    )

    # Event dates provided and correct
    event_dates_leaf = evaluator.add_leaf(
        id="Event_Dates_Provided_And_Correct",
        desc="States the date(s) of the 2025 MAMA Awards event and they are factually correct.",
        parent=top_node,
        critical=True,
    )
    dates_claim = (
        f"The 2025 MAMA Awards event took place on '{dates_text if dates_text else '∅'}'."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=event_dates_leaf,
        sources=source_urls if source_urls else None,
        additional_instruction=(
            "Verify this ONLY using the provided URLs (if any). If no URLs are provided, you must judge the claim as not supported/incorrect. "
            "Accept minor formatting variations of dates (e.g., 'Nov' vs 'November', en-dash vs hyphen, localized formats). "
            "If the event spanned multiple days, accept a range. "
            "Ensure the dates correspond to the 2025 MAMA Awards. "
            "If the extracted dates text is empty, judge as incorrect."
        ),
    )

    # Verifiable sourcing (official/reputable source presence that reports the info)
    verifiable_sourcing_leaf = evaluator.add_leaf(
        id="Verifiable_Sourcing",
        desc="Includes citations/links to official MAMA Awards sources or reputable music industry publications that support the winner, location, and dates stated.",
        parent=top_node,
        critical=True,
    )
    sourcing_claim = (
        "This webpage is either an official MAMA/Mnet site (e.g., mamaawards.com, mnet) or a reputable music industry publication "
        "(e.g., Billboard, Variety, The Hollywood Reporter, Rolling Stone, NME, Soompi). "
        "It also reports the 2025 MAMA Awards 'Song of the Year' winner and event details (winner and at least one of location or dates), "
        "consistent with the answer."
    )
    await evaluator.verify(
        claim=sourcing_claim,
        node=verifiable_sourcing_leaf,
        sources=source_urls if source_urls else None,
        additional_instruction=(
            "Judge each page independently. The page should clearly be from an official MAMA/Mnet domain or a widely recognized music industry outlet. "
            "Additionally, it must report the 2025 MAMA 'Song of the Year' winner and at least one of the event details (location or dates) matching the answer. "
            "If no URLs are provided, judge as not supported/incorrect."
        ),
    )

    # Record a compact custom info snapshot for debugging
    evaluator.add_custom_info(
        info={
            "extracted_song_title": extracted.song_title,
            "extracted_artists": extracted.artists,
            "extracted_location_text": location_text,
            "extracted_dates": dates_text,
            "source_urls_count": len(source_urls),
            "source_urls": source_urls,
        },
        info_type="extraction_snapshot",
        info_name="extraction_snapshot_mama_2025",
    )

    # Return the evaluation summary
    return evaluator.get_summary()