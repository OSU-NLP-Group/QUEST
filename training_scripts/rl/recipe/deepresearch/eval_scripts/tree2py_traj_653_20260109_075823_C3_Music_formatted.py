import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "info_chain_director"
TASK_DESCRIPTION = (
    "Who directed the documentary about the artist who is the younger sister of the youngest person to win the Grammy "
    "Award for Producer of the Year, Non-Classical at the 62nd Annual Grammy Awards in 2020? The artist's debut single "
    "was initially released on SoundCloud in November 2015, and the documentary was released on Apple TV+ on February 26, 2021."
)

# Optional ground truth (for reference in summary only; not used to judge)
GROUND_TRUTH = {
    "producer": {
        "name": "Finneas O'Connell",
        "age_at_winning": "22 years 180 days",
    },
    "artist": {
        "name": "Billie Eilish",
    },
    "debut_single": {
        "title": "Ocean Eyes",
        "platform": "SoundCloud",
        "month_year": "November 2015",
    },
    "documentary": {
        "title": "Billie Eilish: The World's a Little Blurry",
        "platform": "Apple TV+",
        "release_date": "February 26, 2021",
        "director": "R. J. Cutler",
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProducerInfo(BaseModel):
    name: Optional[str] = None
    age_at_winning: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArtistInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DebutSingleInfo(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None  # e.g., "SoundCloud"
    month_year: Optional[str] = None  # e.g., "November 2015"
    sources: List[str] = Field(default_factory=list)


class DocumentaryInfo(BaseModel):
    title: Optional[str] = None
    platform: Optional[str] = None  # e.g., "Apple TV+"
    release_date: Optional[str] = None  # e.g., "February 26, 2021"
    director: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InfoChainExtraction(BaseModel):
    producer: Optional[ProducerInfo] = None
    artist: Optional[ArtistInfo] = None
    debut_single: Optional[DebutSingleInfo] = None
    documentary: Optional[DocumentaryInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_info_chain() -> str:
    return """
    Extract the complete information chain from the provided answer, organized into the following structured fields.
    You must only extract information explicitly present in the answer text. For each sub-entity, also extract any URLs
    the answer cites as sources relevant to that sub-entity.

    Required JSON structure:
    {
      "producer": {
        "name": string | null,                     // The youngest person to win Producer of the Year, Non-Classical at the 62nd Annual Grammy Awards (2020).
        "age_at_winning": string | null,           // The age at the time of winning (e.g., "22 years 180 days").
        "sources": string[]                        // URLs explicitly mentioned that support the producer identity/age claim.
      },
      "artist": {
        "name": string | null,                     // The artist who is the younger sister of the identified producer.
        "sources": string[]                        // URLs supporting the sister relationship / artist identity.
      },
      "debut_single": {
        "title": string | null,                    // The artist's debut single title (if present).
        "platform": string | null,                 // The platform where it was initially released (e.g., "SoundCloud").
        "month_year": string | null,               // The release month and year (e.g., "November 2015").
        "sources": string[]                        // URLs supporting the debut single details.
      },
      "documentary": {
        "title": string | null,                    // The documentary about the artist.
        "platform": string | null,                 // The streaming platform (e.g., "Apple TV+").
        "release_date": string | null,             // The date it was released (e.g., "February 26, 2021").
        "director": string | null,                 // The full name of the director of the documentary.
        "sources": string[]                        // URLs supporting the documentary identity, release details, and director.
      }
    }

    Special URL extraction rules:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - If a URL is missing protocol, prepend http://
    - If no URLs are provided for a field, return an empty array for that field's "sources".
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(extracted: InfoChainExtraction) -> List[str]:
    """Combine all source URLs extracted across sub-entities into a single de-duplicated list."""
    urls: List[str] = []
    if extracted.producer and extracted.producer.sources:
        urls.extend(extracted.producer.sources)
    if extracted.artist and extracted.artist.sources:
        urls.extend(extracted.artist.sources)
    if extracted.debut_single and extracted.debut_single.sources:
        urls.extend(extracted.debut_single.sources)
    if extracted.documentary and extracted.documentary.sources:
        urls.extend(extracted.documentary.sources)
    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u and (u not in seen):
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def safe(val: Optional[str], default: str = "") -> str:
    return val.strip() if isinstance(val, str) else default


# --------------------------------------------------------------------------- #
# Verification tree construction & checks                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_chain(
    evaluator: Evaluator,
    root: Any,
    extracted: InfoChainExtraction
) -> None:
    """
    Build the hierarchical verification tree according to the rubric and run all checks.
    """
    # 0) Create the critical sequential "Complete Information Chain" node
    chain_node = evaluator.add_sequential(
        id="Complete_Information_Chain",
        desc="The answer provides the complete information chain from the Grammy-winning producer to the documentary director.",
        parent=root,
        critical=True
    )

    # Precompute general source fallbacks
    all_sources = combine_sources(extracted)

    # 1) Producer Information (Parallel, Critical)
    producer_node = evaluator.add_parallel(
        id="Producer_Information",
        desc="Correctly identifies the youngest person to win Producer of the Year, Non-Classical at the 62nd Annual Grammy Awards (2020) and provides the required age-at-winning.",
        parent=chain_node,
        critical=True
    )

    producer_name = safe(extracted.producer.name if extracted.producer else None)
    producer_age = safe(extracted.producer.age_at_winning if extracted.producer else None)
    producer_sources = extracted.producer.sources if (extracted.producer and extracted.producer.sources) else all_sources

    # 1.a Producer Identity (Leaf, Critical)
    producer_identity_leaf = evaluator.add_leaf(
        id="Producer_Identity",
        desc="Correctly identifies the producer's name.",
        parent=producer_node,
        critical=True
    )
    producer_identity_claim = (
        f"The youngest person to win the Grammy Award for Producer of the Year, Non-Classical at the 62nd Annual Grammy Awards (2020) is {producer_name}."
        if producer_name else
        "The answer correctly identifies the youngest person to win Producer of the Year, Non-Classical at the 62nd Annual Grammy Awards (2020)."
    )
    await evaluator.verify(
        claim=producer_identity_claim,
        node=producer_identity_leaf,
        sources=producer_sources,
        additional_instruction=(
            "Verify that the named person is explicitly stated as the youngest to win Producer of the Year, Non-Classical "
            "at the 62nd Annual Grammy Awards (2020). Allow minor variations in name formatting."
        )
    )

    # 1.b Producer Age (Leaf, Critical)
    producer_age_leaf = evaluator.add_leaf(
        id="Producer_Age",
        desc="Provides the producer's age at the time of winning (22 years 180 days).",
        parent=producer_node,
        critical=True
    )
    producer_age_claim = (
        f"At the time of winning the award in 2020, {producer_name} was {producer_age}."
        if producer_name and producer_age else
        f"The producer's age at the time of winning the award was {producer_age}."
        if producer_age else
        "The answer provides the producer's age at the time of winning the award."
    )
    await evaluator.verify(
        claim=producer_age_claim,
        node=producer_age_leaf,
        sources=producer_sources,
        additional_instruction=(
            "Confirm the age-at-winning matches the cited sources (e.g., '22 years 180 days'). "
            "Accept reasonable phrasing variants (e.g., '22 years and 180 days')."
        )
    )

    # 2) Artist Identity (Leaf, Critical)
    artist_name = safe(extracted.artist.name if extracted.artist else None)
    artist_sources = extracted.artist.sources if (extracted.artist and extracted.artist.sources) else all_sources

    artist_identity_leaf = evaluator.add_leaf(
        id="Artist_Identity",
        desc="Correctly identifies the artist who is the younger sister of the identified producer.",
        parent=chain_node,
        critical=True
    )
    artist_identity_claim = (
        f"The artist who is the younger sister of {producer_name} is {artist_name}."
        if producer_name and artist_name else
        "The answer correctly identifies the artist who is the younger sister of the identified producer."
    )
    await evaluator.verify(
        claim=artist_identity_claim,
        node=artist_identity_leaf,
        sources=artist_sources,
        additional_instruction=(
            "Verify the sibling relationship and ensure the artist is the younger sister of the identified producer. "
            "Allow minor variations in naming."
        )
    )

    # 3) Artist Verification Facts (Parallel, Critical)
    facts_node = evaluator.add_parallel(
        id="Artist_Verification_Facts",
        desc="Verifies the key facts about the artist and the documentary (debut single release details; documentary release details; director full name).",
        parent=chain_node,
        critical=True
    )

    # 3.a Debut Single Verification (Leaf, Critical)
    debut_platform = safe(extracted.debut_single.platform if extracted.debut_single else None)
    debut_month_year = safe(extracted.debut_single.month_year if extracted.debut_single else None)
    debut_sources = extracted.debut_single.sources if (extracted.debut_single and extracted.debut_single.sources) else all_sources

    debut_leaf = evaluator.add_leaf(
        id="Debut_Single_Verification",
        desc="Correctly states that the artist's debut single was initially released on SoundCloud in November 2015.",
        parent=facts_node,
        critical=True
    )
    # Claim shaped to include artist for clarity; tolerate missing fields
    debut_claim = (
        f"The debut single of {artist_name} was initially released on {debut_platform} in {debut_month_year}."
        if artist_name and debut_platform and debut_month_year else
        "The artist's debut single was initially released on SoundCloud in November 2015."
    )
    await evaluator.verify(
        claim=debut_claim,
        node=debut_leaf,
        sources=debut_sources,
        additional_instruction=(
            "Confirm the initial release platform is SoundCloud and the release month-year is November 2015. "
            "If the source shows exact date in November 2015, that should count as matching 'November 2015'."
        )
    )

    # 3.b Documentary and Director (Parallel, Critical)
    doc_node = evaluator.add_parallel(
        id="Documentary_And_Director",
        desc="Correctly identifies the documentary released on Apple TV+ on February 26, 2021, and provides its director's full name.",
        parent=facts_node,
        critical=True
    )

    doc_title = safe(extracted.documentary.title if extracted.documentary else None)
    doc_platform = safe(extracted.documentary.platform if extracted.documentary else None)
    doc_release = safe(extracted.documentary.release_date if extracted.documentary else None)
    director_name = safe(extracted.documentary.director if extracted.documentary else None)
    doc_sources = extracted.documentary.sources if (extracted.documentary and extracted.documentary.sources) else all_sources

    # 3.b.i Documentary Identity (Leaf, Critical)
    documentary_leaf = evaluator.add_leaf(
        id="Documentary_Identity",
        desc="Correctly identifies the documentary about the artist released on Apple TV+ on February 26, 2021.",
        parent=doc_node,
        critical=True
    )
    if doc_title and artist_name and doc_platform and doc_release:
        doc_claim = (
            f"The documentary about {artist_name} titled '{doc_title}' was released on {doc_platform} on {doc_release}."
        )
    elif artist_name and doc_platform and doc_release:
        doc_claim = (
            f"A documentary about {artist_name} was released on {doc_platform} on {doc_release}."
        )
    else:
        doc_claim = "The documentary was released on Apple TV+ on February 26, 2021."
    await evaluator.verify(
        claim=doc_claim,
        node=documentary_leaf,
        sources=doc_sources,
        additional_instruction=(
            "Verify both the streaming platform (Apple TV+) and the release date (February 26, 2021). "
            "If a title is provided, check that it matches the documentary identified about the artist."
        )
    )

    # 3.b.ii Director Identity (Leaf, Critical)
    director_leaf = evaluator.add_leaf(
        id="Director_Identity",
        desc="Provides the full name of the director who directed the documentary.",
        parent=doc_node,
        critical=True
    )
    if artist_name and doc_platform and doc_release and director_name:
        director_claim = (
            f"The documentary about {artist_name} released on {doc_platform} on {doc_release} was directed by {director_name}."
        )
    elif doc_title and director_name:
        director_claim = f"The documentary '{doc_title}' was directed by {director_name}."
    else:
        director_claim = "The documentary was directed by the named director in the answer."
    await evaluator.verify(
        claim=director_claim,
        node=director_leaf,
        sources=doc_sources,
        additional_instruction=(
            "Verify that the cited director's full name matches the director of the documentary. "
            "Allow reasonable name variants (e.g., initials vs full name)."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the complete information chain (producer → artist → documentary → director).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level can be parallel; the chain node inside is sequential and critical
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

    # Extract structured information chain from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_info_chain(),
        template_class=InfoChainExtraction,
        extraction_name="info_chain_extraction",
    )

    # Add ground truth info (for summary)
    evaluator.add_ground_truth({"expected": GROUND_TRUTH}, gt_type="ground_truth_info_chain")

    # Build and verify the entire chain according to rubric
    await build_and_verify_chain(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()