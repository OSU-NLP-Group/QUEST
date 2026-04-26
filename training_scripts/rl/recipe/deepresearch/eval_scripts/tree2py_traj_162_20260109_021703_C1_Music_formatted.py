import asyncio
import logging
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "producer_birth_year_task"
TASK_DESCRIPTION = """
What is the birth year of a producer who worked on the album that was released in April 2024 and achieved 6x Platinum certification as the Recording Industry Association of America (RIAA)'s top album of 2024?
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AlbumInfo(BaseModel):
    """Information about the qualifying album extracted from the answer."""
    album_name: Optional[str] = None
    release_date_text: Optional[str] = None  # e.g., "April 19, 2024"
    release_month: Optional[str] = None      # e.g., "April"
    release_year: Optional[str] = None       # e.g., "2024"
    sources_release: List[str] = Field(default_factory=list)
    sources_certification: List[str] = Field(default_factory=list)
    sources_riaa_top: List[str] = Field(default_factory=list)


class ProducerInfo(BaseModel):
    """Information about the identified producer extracted from the answer."""
    producer_name: Optional[str] = None
    sources_producer_credit: List[str] = Field(default_factory=list)


class BirthInfo(BaseModel):
    """Birth year information for the identified producer extracted from the answer."""
    birth_year: Optional[str] = None
    sources_birth: List[str] = Field(default_factory=list)


class ProducerBirthYearTaskExtraction(BaseModel):
    """Combined extraction for the entire task."""
    album: Optional[AlbumInfo] = None
    producer: Optional[ProducerInfo] = None
    birth: Optional[BirthInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_task() -> str:
    return """
    Your goal is to extract a single, specific album and a producer from the answer that match the task's constraints, and then provide the producer's birth year with sources.

    Constraints for the album (choose the album that best satisfies all of these as stated in the answer):
    1) The album was released in April 2024.
    2) The album achieved 6x Platinum certification from the Recording Industry Association of America (RIAA).
    3) The album is described or recognized by the RIAA as the top album of 2024 (per the wording in the answer).

    Extraction requirements:
    A. Album information:
       - album_name: The name of the identified album.
       - release_date_text: The textual release date provided (e.g., "April 19, 2024").
       - release_month: The month component of the release date as text (e.g., "April"), if present.
       - release_year: The year component of the release date (e.g., "2024"), if present.
       - sources_release: All URLs cited in the answer that support the release date/month/year of this album.
       - sources_certification: All URLs cited in the answer that support the RIAA certification level (specifically 6x Platinum).
       - sources_riaa_top: All URLs cited in the answer that support the claim the RIAA recognized/described it as the top album of 2024.

    B. Producer information:
       - producer_name: The name of one producer explicitly identified in the answer as having worked on the album.
       - sources_producer_credit: All URLs cited in the answer that support the producer being credited on the album.

    C. Birth year information:
       - birth_year: The birth year stated for the identified producer (e.g., "1984"), as presented in the answer.
       - sources_birth: All URLs cited in the answer that support/confirm the producer's birth year from public biographical sources (e.g., Wikipedia, official bios, credible news outlets).

    General rules:
    - Extract strictly from the provided answer. Do not invent or infer missing data.
    - If any field is missing in the answer, return null for that field or an empty list for sources.
    - Extract only URLs that are explicitly present in the answer (plain URLs or inside markdown links).
    - If multiple albums/producers are mentioned, pick the one that best meets the constraints; if none fully meet them, pick the candidate mentioned and fill missing fields with null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_four_digit_year(text: Optional[str]) -> bool:
    if text is None:
        return False
    text = text.strip()
    return bool(re.fullmatch(r"\d{4}", text))


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_album_constraints(
    evaluator: Evaluator,
    parent_task_node,
    extraction: ProducerBirthYearTaskExtraction,
) -> None:
    """
    Build and verify nodes for "Identify_Qualifying_Album".
    """
    album = extraction.album or AlbumInfo()

    album_node = evaluator.add_parallel(
        id="Identify_Qualifying_Album",
        desc="Identify an album that satisfies the query’s album constraints.",
        parent=parent_task_node,
        critical=True
    )

    # 1) Existence: album named
    album_named = evaluator.add_custom_node(
        result=(album.album_name is not None and album.album_name.strip() != ""),
        id="Album_Identified_By_Name",
        desc="An album is explicitly identified (named) in the answer.",
        parent=album_node,
        critical=True
    )

    # 2) Release in April 2024
    release_leaf = evaluator.add_leaf(
        id="Album_Released_In_April_2024",
        desc="The identified album was released in April 2024.",
        parent=album_node,
        critical=True
    )
    release_claim = f"The album titled '{album.album_name or ''}' was released in April 2024."
    release_sources = album.sources_release if album.sources_release else None
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=release_sources,
        additional_instruction="Verify the album's initial release date was in April 2024. Prefer reliable sources (official site, label page, Wikipedia). Allow region variations but the release month should be April and the year 2024."
    )

    # 3) 6x Platinum certification (RIAA)
    cert_leaf = evaluator.add_leaf(
        id="Album_6x_Platinum_RIAA",
        desc="The identified album achieved 6x Platinum certification from the RIAA.",
        parent=album_node,
        critical=True
    )
    cert_claim = f"The album titled '{album.album_name or ''}' achieved 6x Platinum certification from the Recording Industry Association of America (RIAA)."
    cert_sources = album.sources_certification if album.sources_certification else None
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=cert_sources,
        additional_instruction="Confirm via reliable sources—ideally the RIAA Gold & Platinum database—that the album is certified 6x Platinum."
    )

    # 4) RIAA top album of 2024
    top_leaf = evaluator.add_leaf(
        id="Album_RIAA_Top_Album_Of_2024",
        desc="The identified album is described/recognized by the RIAA as the top album of 2024 (per the question’s constraint wording).",
        parent=album_node,
        critical=True
    )
    top_claim = f"The album titled '{album.album_name or ''}' is recognized by the RIAA as the top album of 2024."
    top_sources = album.sources_riaa_top if album.sources_riaa_top else None
    await evaluator.verify(
        claim=top_claim,
        node=top_leaf,
        sources=top_sources,
        additional_instruction="Verify that the RIAA explicitly recognized/described this album as the 'top album of 2024' (per the wording). Use RIAA publications, press releases, or year-end summaries."
    )


async def verify_producer_credit(
    evaluator: Evaluator,
    parent_task_node,
    extraction: ProducerBirthYearTaskExtraction,
) -> None:
    """
    Build and verify nodes for "Identify_Producer_On_That_Album".
    """
    album_name = (extraction.album.album_name if extraction.album else None) or ""
    producer = extraction.producer or ProducerInfo()

    producer_node = evaluator.add_parallel(
        id="Identify_Producer_On_That_Album",
        desc="Identify a producer who is credited as a producer on the qualifying album.",
        parent=parent_task_node,
        critical=True
    )

    # 1) Producer name existence
    producer_named = evaluator.add_custom_node(
        result=(producer.producer_name is not None and producer.producer_name.strip() != ""),
        id="Producer_Identified_By_Name",
        desc="A producer is explicitly identified (named) in the answer.",
        parent=producer_node,
        critical=True
    )

    # 2) Producer credited on the album
    credit_leaf = evaluator.add_leaf(
        id="Producer_Credited_On_Album",
        desc="The named individual is credited as a producer on the identified qualifying album.",
        parent=producer_node,
        critical=True
    )
    credit_claim = f"{producer.producer_name or ''} is credited as a producer on the album '{album_name}'."
    credit_sources = producer.sources_producer_credit if producer.sources_producer_credit else None
    await evaluator.verify(
        claim=credit_claim,
        node=credit_leaf,
        sources=credit_sources,
        additional_instruction="Confirm producer credit via reliable sources such as official liner notes, label pages, Discogs, Tidal credits, or Wikipedia. Roles like co-producer or additional producer should count."
    )


async def verify_birth_year(
    evaluator: Evaluator,
    parent_task_node,
    extraction: ProducerBirthYearTaskExtraction,
) -> None:
    """
    Build and verify nodes for "Provide_Verifiable_Birth_Year".
    """
    producer_name = (extraction.producer.producer_name if extraction.producer else None) or ""
    birth = extraction.birth or BirthInfo()

    birth_node = evaluator.add_parallel(
        id="Provide_Verifiable_Birth_Year",
        desc="Provide the producer’s birth year, verifiable from public biographical sources.",
        parent=parent_task_node,
        critical=True
    )

    # 1) Birth year stated
    birth_stated = evaluator.add_custom_node(
        result=is_four_digit_year(birth.birth_year),
        id="Birth_Year_Stated",
        desc="The answer states a specific birth year for the identified producer.",
        parent=birth_node,
        critical=True
    )

    # 2) Birth year verifiable via public bio sources
    birth_verify_leaf = evaluator.add_leaf(
        id="Birth_Year_Verifiable_From_Public_Bio_Sources",
        desc="The stated birth year is supported/confirmable via reliable public biographical sources (and matches those sources).",
        parent=birth_node,
        critical=True
    )
    birth_claim = f"The birth year of {producer_name} is {birth.birth_year or ''}."
    birth_sources = birth.sources_birth if birth.sources_birth else None
    await evaluator.verify(
        claim=birth_claim,
        node=birth_verify_leaf,
        sources=birth_sources,
        additional_instruction="Verify the birth year using reliable biographical sources (e.g., Wikipedia, official bios, credible news outlets). The year must match what these sources state."
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
    Evaluate the answer for the Producer Birth Year task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root itself is non-critical; create a critical sequential child
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_task(),
        template_class=ProducerBirthYearTaskExtraction,
        extraction_name="producer_birth_year_task_extraction"
    )

    # Build the top-level critical sequential task node (to enforce ordering and gatekeeping)
    task_node = evaluator.add_sequential(
        id="Producer_Birth_Year_Task",
        desc="Determine the birth year of a producer who worked on an album that meets the stated April 2024 + RIAA 2024 top album + 6x Platinum constraints.",
        parent=root,
        critical=True
    )

    # Run verifications in sequence, respecting the critical gating:
    await verify_album_constraints(evaluator, task_node, extraction)
    await verify_producer_credit(evaluator, task_node, extraction)
    await verify_birth_year(evaluator, task_node, extraction)

    # Add a concise summary of extracted key fields to the output for convenience
    evaluator.add_custom_info(
        info={
            "album_name": (extraction.album.album_name if extraction.album else None),
            "release_date_text": (extraction.album.release_date_text if extraction.album else None),
            "producer_name": (extraction.producer.producer_name if extraction.producer else None),
            "birth_year": (extraction.birth.birth_year if extraction.birth else None),
            "sources_release_count": len(extraction.album.sources_release) if extraction.album else 0,
            "sources_certification_count": len(extraction.album.sources_certification) if extraction.album else 0,
            "sources_riaa_top_count": len(extraction.album.sources_riaa_top) if extraction.album else 0,
            "sources_producer_credit_count": len(extraction.producer.sources_producer_credit) if extraction.producer else 0,
            "sources_birth_count": len(extraction.birth.sources_birth) if extraction.birth else 0
        },
        info_type="extraction_summary",
        info_name="extracted_summary"
    )

    # Return evaluation summary
    return evaluator.get_summary()