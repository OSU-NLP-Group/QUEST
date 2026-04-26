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
TASK_ID = "ska_punk_duration"
TASK_DESCRIPTION = (
    "What is the duration, in years, of the ska-punk band that was founded by the producer of Jimmy Cliff's album "
    "that won the Grammy Award for Best Reggae Album in 2013, before that producer formed the punk band Rancid?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AlbumProducerExtraction(BaseModel):
    """Information about the Grammy-winning album and its producer."""
    album_title: Optional[str] = None
    producer_name: Optional[str] = None
    album_producer_sources: List[str] = Field(default_factory=list)


class BandDurationExtraction(BaseModel):
    """Information about the earlier ska-punk band and its duration."""
    band_name: Optional[str] = None
    formation_date: Optional[str] = None      # e.g., "1987", "May 1987"
    breakup_date: Optional[str] = None        # e.g., "1989", "May 28, 1989"
    duration_years_claim: Optional[str] = None  # e.g., "2 years", "approximately 2"
    band_duration_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_album_producer() -> str:
    return """
    From the answer text, extract the following information related to Jimmy Cliff's Grammy Award for Best Reggae Album in 2013:
    - album_title: The name of the album for which Jimmy Cliff won the Grammy Award for Best Reggae Album in 2013.
    - producer_name: The producer of that Grammy-winning album.
    - album_producer_sources: All URL(s) explicitly provided in the answer that are meant to support the identification of the album and its producer.
    
    Rules:
    - Extract values exactly as they appear in the answer text; do not infer or add new information.
    - For sources, include only actual URLs explicitly present in the answer (plain URLs or URLs inside markdown links). If none are present, return an empty list.
    - If any field is not mentioned, return null (or empty list for sources).
    """


def prompt_extract_band_duration() -> str:
    return """
    From the answer text, extract information about the ska-punk band that the album's producer founded before forming Rancid, and its duration:
    - band_name: The ska-punk band that the producer founded before forming Rancid.
    - formation_date: The formation date/year of that band (e.g., "1987" or "May 1987"). If only a year is present, return that.
    - breakup_date: The breakup date/year of that band (e.g., "1989" or "May 28, 1989"). If only a year is present, return that.
    - duration_years_claim: The duration in years of the band's existence as stated in the answer (e.g., "2 years", "about 2"). If the answer does not explicitly provide a duration, return null.
    - band_duration_sources: All URL(s) explicitly provided in the answer that are meant to support the identification of the band and the formation/breakup timeline (and thus duration).
    
    Rules:
    - Extract values exactly as they appear in the answer text; do not infer or add new information.
    - For sources, include only actual URLs explicitly present in the answer (plain URLs or URLs inside markdown links). If none are present, return an empty list.
    - If any field is not mentioned, return null (or empty list for sources).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    return url.startswith(("http://", "https://")) and "." in url and " " not in url


def extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def parse_duration_years(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Extract the first integer present
    m = re.search(r"\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def compute_duration_years(start_year: Optional[int], end_year: Optional[int]) -> Optional[int]:
    if start_year is None or end_year is None:
        return None
    return max(0, end_year - start_year)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_step1_album_and_producer(
    evaluator: Evaluator,
    parent_node,
    ap: AlbumProducerExtraction
) -> None:
    """
    Step 1: Identify the album that won Jimmy Cliff the Grammy Award for Best Reggae Album in 2013 and its producer;
    ensure supporting URLs are provided.
    Structure:
      - Album_Identification (critical, leaf)
      - Producer_Identification (critical, leaf)
      - URL_Reference_Album_Producer (critical, custom existence check for sources)
    """
    step1_node = evaluator.add_parallel(
        id="Step1_Album_and_Producer",
        desc="Identify the album that won Jimmy Cliff the Grammy Award for Best Reggae Album in 2013 and its producer",
        parent=parent_node,
        critical=False
    )

    # Normalize sources list and filter valid URLs
    valid_album_sources = [u for u in (ap.album_producer_sources or []) if is_valid_url(u)]

    # 1) Album Identification
    album_leaf = evaluator.add_leaf(
        id="Album_Identification",
        desc="Correctly identifies the album that won the Grammy Award for Best Reggae Album in 2013",
        parent=step1_node,
        critical=True
    )
    album_name = ap.album_title or ""
    album_claim = (
        f"Jimmy Cliff won the Grammy Award for Best Reggae Album in 2013 for the album '{album_name}'."
    )
    await evaluator.verify(
        claim=album_claim,
        node=album_leaf,
        sources=valid_album_sources,
        additional_instruction=(
            "Verify that the specified album title corresponds to Jimmy Cliff's award for Best Reggae Album in 2013. "
            "Allow minor variations in title formatting. Use the provided URLs to confirm."
        ),
    )

    # 2) Producer Identification
    producer_leaf = evaluator.add_leaf(
        id="Producer_Identification",
        desc="Correctly identifies the producer of that Grammy-winning album",
        parent=step1_node,
        critical=True
    )
    producer_name = ap.producer_name or ""
    producer_claim = (
        f"The producer of Jimmy Cliff's Grammy-winning album '{album_name}' is '{producer_name}'."
    )
    await evaluator.verify(
        claim=producer_claim,
        node=producer_leaf,
        sources=valid_album_sources,
        additional_instruction=(
            "Verify that the named person is credited as the producer of the specified album. "
            "Allow reasonable title/name variations. Use the provided URLs."
        ),
    )

    # 3) URL Reference existence (make it critical to gate Step 2 in root sequential)
    url_ref_result = len(valid_album_sources) > 0
    evaluator.add_custom_node(
        result=url_ref_result,
        id="URL_Reference_Album_Producer",
        desc="Provides valid URL reference(s) supporting the album and producer identification",
        parent=step1_node,
        critical=True
    )


async def build_step2_band_and_duration(
    evaluator: Evaluator,
    parent_node,
    bd: BandDurationExtraction,
    ap: AlbumProducerExtraction
) -> None:
    """
    Step 2: Identify the ska-punk band founded by the producer before Rancid and calculate its duration.
    Structure (sequential):
      - Earlier_Band_Identification (critical, leaf)
      - Duration_Calculation (critical, leaf)
      - URL_Reference_Band_Duration (critical, custom existence check for sources)
    """
    step2_node = evaluator.add_sequential(
        id="Step2_Band_and_Duration",
        desc="Identify the ska-punk band founded by the producer before Rancid and calculate its duration",
        parent=parent_node,
        critical=False
    )

    valid_band_sources = [u for u in (bd.band_duration_sources or []) if is_valid_url(u)]
    producer_name = ap.producer_name or ""
    band_name = bd.band_name or ""

    # 1) Earlier Band Identification
    band_leaf = evaluator.add_leaf(
        id="Earlier_Band_Identification",
        desc="Correctly identifies the ska-punk band that the producer founded before forming Rancid",
        parent=step2_node,
        critical=True
    )
    band_claim = (
        f"The ska-punk band that {producer_name} founded before forming Rancid is '{band_name}'."
    )
    await evaluator.verify(
        claim=band_claim,
        node=band_leaf,
        sources=valid_band_sources,
        additional_instruction=(
            "Confirm that the named band was founded (or co-founded) by the specified producer prior to the formation of Rancid. "
            "Use the provided URLs to verify."
        ),
    )

    # 2) Duration Calculation
    duration_leaf = evaluator.add_leaf(
        id="Duration_Calculation",
        desc="Correctly calculates the duration of the band's existence from formation to breakup, expressed in years",
        parent=step2_node,
        critical=True
    )

    start_year = extract_year(bd.formation_date)
    end_year = extract_year(bd.breakup_date)
    claimed_duration = parse_duration_years(bd.duration_years_claim)
    computed_duration = compute_duration_years(start_year, end_year)

    # Prefer an explicit claim from the answer; otherwise use computed years if available.
    if claimed_duration is not None:
        duration_to_check = claimed_duration
    else:
        duration_to_check = computed_duration if computed_duration is not None else None

    # Build a claim that references the dates and the duration
    if duration_to_check is None:
        duration_claim = (
            f"The band '{band_name}' existed from '{bd.formation_date or ''}' to '{bd.breakup_date or ''}', "
            f"which corresponds to an integer number of years."
        )
        add_ins = (
            "Using the provided URLs, check the formation and breakup years of the band, and determine the integer-year duration. "
            "If the dates are missing or inconsistent, judge the statement as incorrect."
        )
    else:
        duration_claim = (
            f"The band '{band_name}' existed from '{bd.formation_date or ''}' to '{bd.breakup_date or ''}', "
            f"which corresponds to approximately {duration_to_check} years."
        )
        add_ins = (
            "Verify the formation and breakup timeline using the provided URLs, and confirm that the integer-year difference "
            f"matches {duration_to_check}. Allow minor rounding when months/days are involved."
        )

    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=valid_band_sources,
        additional_instruction=add_ins,
    )

    # 3) URL Reference existence for band/duration
    url_ref_band_result = len(valid_band_sources) > 0
    evaluator.add_custom_node(
        result=url_ref_band_result,
        id="URL_Reference_Band_Duration",
        desc="Provides valid URL reference(s) supporting the band identification and duration calculation",
        parent=step2_node,
        critical=True
    )

    # Record helpful computation details for debugging in summary
    evaluator.add_custom_info(
        info={
            "band_name": band_name,
            "producer_name": producer_name,
            "formation_date_extracted": bd.formation_date,
            "breakup_date_extracted": bd.breakup_date,
            "claimed_duration_years_extracted": bd.duration_years_claim,
            "parsed_start_year": start_year,
            "parsed_end_year": end_year,
            "computed_duration_years": computed_duration,
            "duration_used_for_verification": duration_to_check
        },
        info_type="computation",
        info_name="band_duration_computation"
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
    Evaluate an answer for the ska-punk duration task.

    Returns:
        A structured summary dictionary produced by the evaluator.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Follow rubric: Step2 depends on Step1
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

    # Extract required information (can run concurrently)
    ap_task = evaluator.extract(
        prompt=prompt_extract_album_producer(),
        template_class=AlbumProducerExtraction,
        extraction_name="album_producer_extraction"
    )
    bd_task = evaluator.extract(
        prompt=prompt_extract_band_duration(),
        template_class=BandDurationExtraction,
        extraction_name="band_duration_extraction"
    )
    ap, bd = await asyncio.gather(ap_task, bd_task)

    # Build verification steps
    await build_step1_album_and_producer(evaluator, root, ap)
    await build_step2_band_and_duration(evaluator, root, bd, ap)

    # Return evaluation summary
    return evaluator.get_summary()