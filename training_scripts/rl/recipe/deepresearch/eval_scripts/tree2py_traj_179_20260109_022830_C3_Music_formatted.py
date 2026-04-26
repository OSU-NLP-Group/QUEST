import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "album_producer_studio_artist_task"
TASK_DESCRIPTION = (
    "A female artist who is the vocalist and bassist of a garage rock band released her solo debut album in 2018. "
    "The album title contains the word 'Nashville' and was produced by a Grammy-winning producer who is a member of The Black Keys. "
    "The album was recorded at the producer's own studio in Nashville, Tennessee, located on 8th Avenue South. "
    "Provide the following information: (1) The full title of the album, (2) The name of the producer, "
    "(3) The complete address of the recording studio, (4) The month and year when the producer purchased the studio property, "
    "(5) The artist's full birth date (day, month, year), and (6) The city in California where the artist was born. "
    "All answers must be supported by reference URLs."
)


# =========================
# Data Models
# =========================
class TaskExtraction(BaseModel):
    # Core entities
    album_title: Optional[str] = None
    producer_name: Optional[str] = None
    studio_name: Optional[str] = None
    studio_address: Optional[str] = None
    studio_purchase_month_year: Optional[str] = None  # e.g., "May 2017"
    artist_name: Optional[str] = None
    artist_birth_date: Optional[str] = None  # e.g., "June 19, 1986"
    artist_birth_city: Optional[str] = None  # e.g., "Napa" (must be in California)
    band_name: Optional[str] = None  # e.g., "Shannon and the Clams"

    # Source URLs grouped by topic
    album_sources: List[str] = Field(default_factory=list)
    producer_sources: List[str] = Field(default_factory=list)
    studio_sources: List[str] = Field(default_factory=list)
    artist_sources: List[str] = Field(default_factory=list)


# =========================
# Extraction Prompt
# =========================
def prompt_extract_all() -> str:
    return """
    Extract the following structured information strictly from the provided answer text. If any item is not present in the answer, return null for that field. 
    Additionally, extract URL sources explicitly mentioned in the answer for each category (album, producer, studio, artist). Follow the URL extraction rules.

    Required fields:
    - album_title: The exact full title of the album.
    - producer_name: The producer's full name.
    - studio_name: The name of the producer's recording studio (if given).
    - studio_address: The complete address of the recording studio (street number/name + city + state + ZIP if present).
    - studio_purchase_month_year: The month and year when the producer purchased the studio property (e.g., "May 2017").
    - artist_name: The artist’s full name.
    - artist_birth_date: The artist's full birth date (day, month, year), as stated in the answer.
    - artist_birth_city: The city in California where the artist was born, exactly as stated in the answer.
    - band_name: The garage rock band for which the artist is the vocalist and bassist (if mentioned explicitly).

    URL Sources (must be explicit URLs in the answer; do not invent):
    - album_sources: URLs supporting album-related facts (title, release year 2018, solo debut, recorded at producer’s studio).
    - producer_sources: URLs supporting producer-related facts (Grammy-winning, member of The Black Keys, producer of the album).
    - studio_sources: URLs supporting studio-related facts (recording location in Nashville on 8th Avenue South, the full address, purchase month/year).
    - artist_sources: URLs supporting artist-related facts (female, vocalist and bassist of a garage rock band, birth date, CA birth city).

    Return a JSON object with exactly these fields.
    """


# =========================
# Helpers
# =========================
def _is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def _filter_valid_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls or []:
        if _is_valid_url(u):
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _merge_sources(*lists: List[str]) -> List[str]:
    merged = []
    seen = set()
    for lst in lists:
        for u in lst or []:
            if _is_valid_url(u) and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _contains_nashville(title: Optional[str]) -> bool:
    if not title:
        return False
    return "nashville" in title.lower()


# =========================
# Tree Builders (Verification)
# =========================
async def build_album_branch(evaluator: Evaluator, parent, ex: TaskExtraction) -> None:
    album_node = evaluator.add_parallel(
        id="Album_Identification_And_Constraints",
        desc="Identify the album and verify album-related constraints.",
        parent=parent,
        critical=True
    )

    # Existence: Full title provided
    evaluator.add_custom_node(
        result=bool(ex.album_title and ex.album_title.strip()),
        id="Album_Full_Title_Provided",
        desc="Provides the full title of the album.",
        parent=album_node,
        critical=True
    )

    # Title contains 'Nashville' (string check)
    evaluator.add_custom_node(
        result=_contains_nashville(ex.album_title),
        id="Album_Title_Contains_Nashville",
        desc="Verifies the album title contains the word 'Nashville'.",
        parent=album_node,
        critical=True
    )

    # Album released in 2018 (verify via URLs)
    rel_2018_node = evaluator.add_leaf(
        id="Album_Released_In_2018",
        desc="Verifies the album was released in 2018.",
        parent=album_node,
        critical=True
    )
    claim_release_2018 = f"The album '{ex.album_title or ''}' was released in 2018."
    await evaluator.verify(
        claim=claim_release_2018,
        node=rel_2018_node,
        sources=_filter_valid_urls(ex.album_sources),
        additional_instruction="Look for the album release date/year on the provided page(s) and confirm it was in 2018."
    )

    # Solo debut album (verify via URLs; union of album + artist sources)
    solo_debut_node = evaluator.add_leaf(
        id="Album_Is_Solo_Debut",
        desc="Verifies the album is the artist's solo debut album.",
        parent=album_node,
        critical=True
    )
    claim_solo_debut = f"The album '{ex.album_title or ''}' is the solo debut album of the artist {ex.artist_name or ''}."
    await evaluator.verify(
        claim=claim_solo_debut,
        node=solo_debut_node,
        sources=_merge_sources(_filter_valid_urls(ex.album_sources), _filter_valid_urls(ex.artist_sources)),
        additional_instruction="Confirm the album is described as the artist's 'solo debut album' or equivalent phrasing."
    )


async def build_artist_branch(evaluator: Evaluator, parent, ex: TaskExtraction) -> None:
    artist_node = evaluator.add_parallel(
        id="Artist_Constraints_And_Bio",
        desc="Verify the artist matches the described constraints and provide requested biographical details.",
        parent=parent,
        critical=True
    )

    # Female
    is_female_node = evaluator.add_leaf(
        id="Artist_Is_Female",
        desc="Verifies the artist is female.",
        parent=artist_node,
        critical=True
    )
    claim_female = f"The artist {ex.artist_name or ''} is female."
    await evaluator.verify(
        claim=claim_female,
        node=is_female_node,
        sources=_filter_valid_urls(ex.artist_sources),
        additional_instruction="Use the provided biography/reference(s) to confirm the artist is female (e.g., pronouns like 'she', or explicit statements)."
    )

    # Vocalist and bassist of a garage rock band
    vb_node = evaluator.add_leaf(
        id="Artist_Is_Vocalist_And_Bassist_Of_Garage_Rock_Band",
        desc="Verifies the artist is the vocalist and bassist of a garage rock band.",
        parent=artist_node,
        critical=True
    )
    if ex.band_name:
        claim_vb = f"{ex.artist_name or ''} is the vocalist and bassist of the garage rock band {ex.band_name}."
    else:
        claim_vb = f"{ex.artist_name or ''} is the vocalist and bassist of a garage rock band."
    await evaluator.verify(
        claim=claim_vb,
        node=vb_node,
        sources=_filter_valid_urls(ex.artist_sources),
        additional_instruction="Confirm both roles (vocalist and bassist) and the genre 'garage rock' (allow close variants like garage punk/garage soul/garage pop if the band is broadly categorized as garage rock)."
    )

    # Birth date provided (existence)
    evaluator.add_custom_node(
        result=bool(ex.artist_birth_date and ex.artist_birth_date.strip()),
        id="Artist_Full_Birth_Date_Provided",
        desc="Provides the artist's full birth date (day, month, year).",
        parent=artist_node,
        critical=True
    )

    # Birth city provided (existence)
    evaluator.add_custom_node(
        result=bool(ex.artist_birth_city and ex.artist_birth_city.strip()),
        id="Artist_Born_In_California_City_Provided",
        desc="Provides the city in California where the artist was born.",
        parent=artist_node,
        critical=True
    )


async def build_producer_studio_branch(evaluator: Evaluator, parent, ex: TaskExtraction) -> None:
    prod_studio_node = evaluator.add_parallel(
        id="Producer_And_Recording_Studio",
        desc="Identify the producer, verify producer/studio constraints, and provide studio/address/purchase info.",
        parent=parent,
        critical=True
    )

    # Producer name provided
    evaluator.add_custom_node(
        result=bool(ex.producer_name and ex.producer_name.strip()),
        id="Producer_Name_Provided",
        desc="Provides the producer’s name.",
        parent=prod_studio_node,
        critical=True
    )

    # Producer is Grammy-winning
    grammy_node = evaluator.add_leaf(
        id="Producer_Is_Grammy_Winning",
        desc="Verifies the producer is Grammy-winning.",
        parent=prod_studio_node,
        critical=True
    )
    claim_grammy = f"The producer {ex.producer_name or ''} is a Grammy-winning musician/producer."
    await evaluator.verify(
        claim=claim_grammy,
        node=grammy_node,
        sources=_filter_valid_urls(ex.producer_sources),
        additional_instruction="Confirm the producer has won a Grammy Award (not just nominated)."
    )

    # Producer is a member of The Black Keys
    bk_node = evaluator.add_leaf(
        id="Producer_Is_Black_Keys_Member",
        desc="Verifies the producer is a member of The Black Keys.",
        parent=prod_studio_node,
        critical=True
    )
    claim_bk = f"The producer {ex.producer_name or ''} is a member of The Black Keys."
    await evaluator.verify(
        claim=claim_bk,
        node=bk_node,
        sources=_filter_valid_urls(ex.producer_sources),
        additional_instruction="Confirm that the producer is/was a member (e.g., co-founder, guitarist/vocalist) of The Black Keys."
    )

    # Album produced by producer
    produced_by_node = evaluator.add_leaf(
        id="Album_Produced_By_Producer",
        desc="Verifies the identified album was produced by the identified producer.",
        parent=prod_studio_node,
        critical=True
    )
    claim_produced = f"The album '{ex.album_title or ''}' was produced by {ex.producer_name or ''}."
    await evaluator.verify(
        claim=claim_produced,
        node=produced_by_node,
        sources=_merge_sources(_filter_valid_urls(ex.album_sources), _filter_valid_urls(ex.producer_sources)),
        additional_instruction="Confirm producer credit on the album page or producer's references."
    )

    # Album recorded at producer's own studio in Nashville, on 8th Avenue South
    recorded_node = evaluator.add_leaf(
        id="Album_Recorded_At_Producer_Own_Studio_In_Nashville_On_8th_Ave_South",
        desc="Verifies the album was recorded at the producer's own studio in Nashville, Tennessee, located on 8th Avenue South.",
        parent=prod_studio_node,
        critical=True
    )
    studio_label = ex.studio_name or "the producer's own studio"
    claim_recorded = (
        f"The album '{ex.album_title or ''}' was recorded at {studio_label}, the producer {ex.producer_name or ''}'s own studio, "
        f"in Nashville, Tennessee, which is located on 8th Avenue South."
    )
    await evaluator.verify(
        claim=claim_recorded,
        node=recorded_node,
        sources=_merge_sources(_filter_valid_urls(ex.album_sources), _filter_valid_urls(ex.studio_sources)),
        additional_instruction="Verify both ownership (producer's own studio), Nashville location, and that the studio is on 8th Avenue South. The page(s) should support these points."
    )

    # Studio complete address provided (existence)
    evaluator.add_custom_node(
        result=bool(ex.studio_address and ex.studio_address.strip()),
        id="Studio_Complete_Address_Provided",
        desc="Provides the complete address of the recording studio.",
        parent=prod_studio_node,
        critical=True
    )

    # Studio purchase month/year provided (existence)
    evaluator.add_custom_node(
        result=bool(ex.studio_purchase_month_year and ex.studio_purchase_month_year.strip()),
        id="Studio_Property_Purchase_Month_And_Year_Provided",
        desc="Provides the month and year when the producer purchased the studio property.",
        parent=prod_studio_node,
        critical=True
    )


def build_citations_branch(evaluator: Evaluator, parent, ex: TaskExtraction) -> None:
    citations_node = evaluator.add_parallel(
        id="Citations",
        desc="All provided answers and constraint verifications include supporting reference URLs.",
        parent=parent,
        critical=True
    )

    album_urls = _filter_valid_urls(ex.album_sources)
    producer_urls = _filter_valid_urls(ex.producer_sources)
    studio_urls = _filter_valid_urls(ex.studio_sources)
    artist_urls = _filter_valid_urls(ex.artist_sources)

    # All four groups must have at least one valid URL to support the provided outputs and constraints
    all_supported = bool(album_urls) and bool(producer_urls) and bool(studio_urls) and bool(artist_urls)

    evaluator.add_custom_node(
        result=all_supported,
        id="Reference_URLs_Provided_For_All_Claims",
        desc="Includes reference URL(s) that support each requested output field and the key constraint verifications used to identify the correct album/producer/studio/artist.",
        parent=citations_node,
        critical=True
    )


# =========================
# Main Evaluation
# =========================
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
        default_model=model
    )

    # Extract all structured info
    ex: TaskExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=TaskExtraction,
        extraction_name="extracted_fields"
    )

    # Build Task_Completion node (critical)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Provide all requested album/producer/studio/artist information, satisfy all stated constraints, and include supporting reference URLs.",
        parent=root,
        critical=True
    )

    # Build sub-branches
    await build_album_branch(evaluator, task_node, ex)
    await build_artist_branch(evaluator, task_node, ex)
    await build_producer_studio_branch(evaluator, task_node, ex)
    build_citations_branch(evaluator, task_node, ex)

    # Return evaluation summary
    return evaluator.get_summary()