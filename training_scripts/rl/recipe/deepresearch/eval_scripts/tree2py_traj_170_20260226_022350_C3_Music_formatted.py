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
TASK_ID = "producer_2023_debut_release_month_year"
TASK_DESCRIPTION = (
    "A music producer won Producer of the Year (Non-Classical) at the 67th Annual Grammy Awards ceremony "
    "that took place on February 2, 2025. This producer was previously a member of an indie rock band that was "
    "formed between 2001 and 2002 in Long Island, New York. The producer also worked on an album that won Best Pop Vocal "
    "Album at the 2022 Grammy Awards; that winning album was released in 2021. Additionally, the same producer worked on "
    "another artist's debut studio album that was released in 2023 and was nominated for Album of the Year at the 67th Annual "
    "Grammy Awards (the 2025 ceremony). What is the release month and year of this 2023 debut studio album?"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProducerAlbumExtraction(BaseModel):
    # Producer identification and award
    producer_name: Optional[str] = None
    award_source_urls: List[str] = Field(default_factory=list)

    # Indie rock band information
    indie_band_name: Optional[str] = None
    indie_band_formation_year_range: Optional[str] = None  # e.g., "2001-2002" or "formed in 2001/2002"
    indie_band_location: Optional[str] = None  # e.g., "Long Island, New York"
    indie_band_source_urls: List[str] = Field(default_factory=list)

    # 2022 Best Pop Vocal Album project details (the album producer worked on)
    bpva_album_title: Optional[str] = None
    bpva_album_artist: Optional[str] = None
    bpva_album_release_year: Optional[str] = None  # should be "2021"
    bpva_album_source_urls: List[str] = Field(default_factory=list)

    # 2023 debut studio album details (the target for the final question)
    debut_album_title: Optional[str] = None
    debut_album_artist: Optional[str] = None
    debut_album_release_month: Optional[str] = None  # e.g., "September"
    debut_album_release_year: Optional[str] = None   # e.g., "2023"
    debut_album_source_urls: List[str] = Field(default_factory=list)

    # Nomination evidence (Album of the Year at 67th Grammys)
    nomination_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_producer_and_album_info() -> str:
    return """
    Extract the structured information from the answer needed to verify the constraints and identify the release month and year.

    Return a single JSON object with the following fields (strings unless otherwise noted):
    - producer_name: The full name of the producer who satisfies all constraints.
    - award_source_urls: Array of URL strings explicitly cited in the answer that substantiate the claim that the producer won Producer of the Year (Non-Classical) at the 67th Annual Grammy Awards held on February 2, 2025.
    - indie_band_name: The name of the indie rock band the producer was previously a member of.
    - indie_band_formation_year_range: The formation year or range (e.g., "2001", "2002", "2001-2002", "formed in 2001/2002").
    - indie_band_location: The location of the band’s formation (e.g., "Long Island, New York").
    - indie_band_source_urls: Array of URL strings explicitly cited in the answer that support the band membership and the band's formation details (year 2001/2002 and location Long Island, NY).
    - bpva_album_title: The title of the album the producer worked on which won Best Pop Vocal Album at the 2022 Grammys.
    - bpva_album_artist: The primary artist of that album.
    - bpva_album_release_year: The release year of that album (expected to be "2021").
    - bpva_album_source_urls: Array of URL strings cited in the answer that support the producer's involvement, the award win (Best Pop Vocal Album at the 2022 Grammys), and the release year (2021).
    - debut_album_title: The title of the 2023 debut studio album (by another artist) that the producer worked on.
    - debut_album_artist: The artist of that 2023 debut studio album.
    - debut_album_release_month: The release month (e.g., "September") of that 2023 debut studio album.
    - debut_album_release_year: The release year (should be "2023") of that 2023 debut studio album.
    - debut_album_source_urls: Array of URL strings cited in the answer that support the producer's involvement and the album's details (debut status and release).
    - nomination_source_urls: Array of URL strings cited in the answer that support the nomination for Album of the Year at the 67th Annual Grammy Awards (the 2025 ceremony).

    RULES:
    - Extract only information explicitly present in the answer. Do not invent.
    - For URL fields, return only valid URLs (plain urls or urls inside markdown links). If missing, return an empty array.
    - If a string field is not present, return null.
    - Prefer the most precise wording the answer provides (e.g., month names like "September" rather than abbreviations), but do not invent details.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_urls(*url_lists: List[str]) -> List[str]:
    """Combine multiple URL lists and de-duplicate, filtering out blank entries."""
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            u2 = u.strip()
            if not u2:
                continue
            if u2 not in seen:
                seen.add(u2)
                combined.append(u2)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_producer_constraints(
    evaluator: Evaluator,
    parent_node,
    data: ProducerAlbumExtraction,
) -> None:
    """
    Build and verify the subtree:
      Identify_Producer_Meeting_All_Producer_Constraints (parallel, all critical)
        ├─ Won_Producer_of_the_Year_Non_Classical_At_67th_Grammys (leaf)
        ├─ Member_Of_Indie_Rock_Band_Formed_2001_or_2002_in_Long_Island_NY (leaf)
        └─ Verify_2022_Best_Pop_Vocal_Album_Project (parallel)
            ├─ Producer_Worked_On_Target_Best_Pop_Vocal_Album_Project (leaf)
            ├─ Target_Album_Won_Best_Pop_Vocal_Album_At_2022_Grammys (leaf)
            └─ Target_Album_Released_In_2021 (leaf)
    """
    # Create the producer constraints node (critical parallel)
    producer_node = evaluator.add_parallel(
        id="Identify_Producer_Meeting_All_Producer_Constraints",
        desc="Identify a music producer who satisfies all producer-related constraints from the prompt.",
        parent=parent_node,
        critical=True
    )

    # 1) Award winner verification
    award_leaf = evaluator.add_leaf(
        id="Won_Producer_of_the_Year_Non_Classical_At_67th_Grammys",
        desc="Producer won Producer of the Year (Non-Classical) at the 67th Annual Grammy Awards held on February 2, 2025.",
        parent=producer_node,
        critical=True
    )
    award_claim = (
        f"{data.producer_name or ''} won Producer of the Year (Non-Classical) at the 67th Annual Grammy Awards, "
        f"which took place on February 2, 2025."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_leaf,
        sources=data.award_source_urls,
        additional_instruction=(
            "Confirm both: (1) the producer's win in the 'Producer of the Year (Non-Classical)' category, "
            "and (2) that this occurred at the 67th Annual Grammy Awards (held on Feb 2, 2025). "
            "Use the provided URLs (e.g., official Grammys site, reputable outlets, or Wikipedia) to support the claim."
        )
    )

    # 2) Indie band membership and formation details
    band_leaf = evaluator.add_leaf(
        id="Member_Of_Indie_Rock_Band_Formed_2001_or_2002_in_Long_Island_NY",
        desc="Producer was previously a member of an indie rock band formed in 2001 or 2002 in Long Island, New York.",
        parent=producer_node,
        critical=True
    )
    band_claim = (
        f"{data.producer_name or ''} was previously a member of the indie rock band {data.indie_band_name or ''}, "
        f"which was formed in 2001 or 2002 in Long Island, New York."
    )
    await evaluator.verify(
        claim=band_claim,
        node=band_leaf,
        sources=data.indie_band_source_urls,
        additional_instruction=(
            "Verify the producer's membership in the named indie rock band, and that the band's formation aligns with "
            "the constraint (formed in 2001 or 2002) and the location (Long Island, New York). Allow common phrasing variants."
        )
    )

    # 3) 2022 Best Pop Vocal Album project (parallel, all critical)
    bpva_node = evaluator.add_parallel(
        id="Verify_2022_Best_Pop_Vocal_Album_Project",
        desc="Verify there exists a single album project such that the producer worked on it, it won Best Pop Vocal Album at the 2022 Grammys, and it was released in 2021.",
        parent=producer_node,
        critical=True
    )

    # 3.1) Producer worked on the target album
    bpva_work_leaf = evaluator.add_leaf(
        id="Producer_Worked_On_Target_Best_Pop_Vocal_Album_Project",
        desc="Producer worked on the album that is being used to satisfy the 2022 Best Pop Vocal Album constraint.",
        parent=bpva_node,
        critical=True
    )
    bpva_work_claim = (
        f"{data.producer_name or ''} worked on the album '{data.bpva_album_title or ''}' by {data.bpva_album_artist or ''}."
    )
    await evaluator.verify(
        claim=bpva_work_claim,
        node=bpva_work_leaf,
        sources=data.bpva_album_source_urls,
        additional_instruction=(
            "Confirm the producer's involvement (e.g., producer, co-producer, production, engineering, significant credit) "
            "with the specified album using the provided URLs."
        )
    )

    # 3.2) Target album won Best Pop Vocal Album at the 2022 Grammys
    bpva_award_leaf = evaluator.add_leaf(
        id="Target_Album_Won_Best_Pop_Vocal_Album_At_2022_Grammys",
        desc="That same target album won Best Pop Vocal Album at the 2022 Grammy Awards.",
        parent=bpva_node,
        critical=True
    )
    bpva_award_claim = (
        f"The album '{data.bpva_album_title or ''}' by {data.bpva_album_artist or ''} won Best Pop Vocal Album at the 2022 Grammy Awards."
    )
    await evaluator.verify(
        claim=bpva_award_claim,
        node=bpva_award_leaf,
        sources=data.bpva_album_source_urls,
        additional_instruction=(
            "Confirm the category 'Best Pop Vocal Album' and the award year (the ceremony held in 2022). "
            "Use credible sources like official Grammys pages or Wikipedia."
        )
    )

    # 3.3) Target album released in 2021
    bpva_release_leaf = evaluator.add_leaf(
        id="Target_Album_Released_In_2021",
        desc="That same target album was released in 2021.",
        parent=bpva_node,
        critical=True
    )
    bpva_release_claim = (
        f"The album '{data.bpva_album_title or ''}' by {data.bpva_album_artist or ''} was released in 2021."
    )
    await evaluator.verify(
        claim=bpva_release_claim,
        node=bpva_release_leaf,
        sources=data.bpva_album_source_urls,
        additional_instruction=(
            "Confirm the album's original release year is 2021 (month/day can vary). Prefer official sources or reputable discography references."
        )
    )


async def verify_debut_album_constraints(
    evaluator: Evaluator,
    parent_node,
    data: ProducerAlbumExtraction,
) -> None:
    """
    Build and verify the subtree:
      Identify_Qualifying_2023_Debut_Studio_Album (parallel, all critical)
        ├─ Producer_Worked_On_Debut_Studio_Album_Released_In_2023 (leaf)
        └─ Target_Album_Nominated_For_Album_of_the_Year_At_67th_Grammys (leaf)
    """
    debut_node = evaluator.add_parallel(
        id="Identify_Qualifying_2023_Debut_Studio_Album",
        desc="Identify the 2023 debut studio album (by another artist) that the producer worked on and that meets the Grammy nomination constraint.",
        parent=parent_node,
        critical=True
    )

    # 1) Producer worked on debut studio album released in 2023
    debut_work_leaf = evaluator.add_leaf(
        id="Producer_Worked_On_Debut_Studio_Album_Released_In_2023",
        desc="Producer worked on another artist's debut studio album that was released in 2023.",
        parent=debut_node,
        critical=True
    )
    debut_work_claim = (
        f"{data.producer_name or ''} worked on the debut studio album '{data.debut_album_title or ''}' by {data.debut_album_artist or ''}, "
        f"which was released in 2023."
    )
    await evaluator.verify(
        claim=debut_work_claim,
        node=debut_work_leaf,
        sources=data.debut_album_source_urls,
        additional_instruction=(
            "Confirm both: (1) the producer's involvement (producer/co-producer/production/engineering/major credit) and "
            "(2) that the album is the artist's debut studio album and was released in 2023."
        )
    )

    # 2) Target album nominated for Album of the Year at the 67th Grammys
    nomination_leaf = evaluator.add_leaf(
        id="Target_Album_Nominated_For_Album_of_the_Year_At_67th_Grammys",
        desc="The target 2023 debut studio album was nominated for Album of the Year at the 67th Annual Grammy Awards (2025 ceremony).",
        parent=debut_node,
        critical=True
    )
    combined_nomination_sources = _combine_urls(data.debut_album_source_urls, data.nomination_source_urls)
    nomination_claim = (
        f"The album '{data.debut_album_title or ''}' by {data.debut_album_artist or ''} was nominated for Album of the Year "
        f"at the 67th Annual Grammy Awards (2025 ceremony)."
    )
    await evaluator.verify(
        claim=nomination_claim,
        node=nomination_leaf,
        sources=combined_nomination_sources,
        additional_instruction=(
            "Confirm this album appears among the nominees for 'Album of the Year' at the 67th Grammys (2025 ceremony). "
            "Use official Grammys pages, reputable outlets, or comprehensive references like Wikipedia."
        )
    )


async def verify_release_month_year(
    evaluator: Evaluator,
    parent_node,
    data: ProducerAlbumExtraction,
) -> None:
    """
    Build and verify the leaf:
      Provide_Release_Month_And_Year (leaf, critical)
    """
    release_leaf = evaluator.add_leaf(
        id="Provide_Release_Month_And_Year",
        desc="Provide the release month and year of the identified 2023 debut studio album.",
        parent=parent_node,
        critical=True
    )

    release_claim = (
        f"The album '{data.debut_album_title or ''}' by {data.debut_album_artist or ''} was released in "
        f"{data.debut_album_release_month or ''} {data.debut_album_release_year or ''}."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=data.debut_album_source_urls,
        additional_instruction=(
            "Verify the album's release month and year. Accept common month variants or abbreviations (e.g., 'Sept.' = 'September'). "
            "Prefer official announcements, label pages, or reputable discography sources."
        )
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
    Evaluate the answer for the producer/album constraints and verify the release month/year.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Top-level logical order
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

    # Create the main critical sequential node under root (to reflect rubric root)
    main_node = evaluator.add_sequential(
        id="Identify_2023_Debut_Album_Release_Month_Year",
        desc="Identify the producer satisfying all constraints, identify the relevant 2023 debut studio album, and provide its release month and year.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_producer_and_album_info(),
        template_class=ProducerAlbumExtraction,
        extraction_name="producer_and_album_info",
    )

    # Build verification tree according to rubric
    await verify_producer_constraints(evaluator, main_node, extracted)
    await verify_debut_album_constraints(evaluator, main_node, extracted)
    await verify_release_month_year(evaluator, main_node, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()