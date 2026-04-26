import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grammy_2026_producer_2025_albums"
TASK_DESCRIPTION = (
    "For each of the five 2026 Grammy Award nominees for Producer of the Year, Non-Classical "
    "(Dan Auerbach, Cirkut, Dijon, Blake Mills, and Sounwave), identify one album released in 2025 "
    "that the producer worked on for another primary artist (not their own solo work). For each album, "
    "provide the artist name, album title, and a reference URL that verifies the producer's credit and "
    "the album's 2025 release date."
)

PRODUCERS: List[Tuple[str, str, str]] = [
    ("dan_auerbach", "Dan Auerbach", "Producer_1"),
    ("cirkut", "Cirkut", "Producer_2"),
    ("dijon", "Dijon", "Producer_3"),
    ("blake_mills", "Blake Mills", "Producer_4"),
    ("sounwave", "Sounwave", "Producer_5"),
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProducerAlbumItem(BaseModel):
    """
    One album entry for a specific producer.
    """
    producer: Optional[str] = None
    primary_artist: Optional[str] = None
    album_title: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class AllProducersExtraction(BaseModel):
    """
    Extract exactly one album entry per specified producer.
    """
    dan_auerbach: ProducerAlbumItem = Field(default_factory=ProducerAlbumItem)
    cirkut: ProducerAlbumItem = Field(default_factory=ProducerAlbumItem)
    dijon: ProducerAlbumItem = Field(default_factory=ProducerAlbumItem)
    blake_mills: ProducerAlbumItem = Field(default_factory=ProducerAlbumItem)
    sounwave: ProducerAlbumItem = Field(default_factory=ProducerAlbumItem)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_producer_albums() -> str:
    """
    Instruct the LLM to extract one album per producer with fields required for verification.
    """
    return """
    Extract exactly one album entry for each of the following producers: Dan Auerbach, Cirkut, Dijon, Blake Mills, and Sounwave.
    For each producer, extract:
    - producer: the producer's name (must be exactly one of: "Dan Auerbach", "Cirkut", "Dijon", "Blake Mills", "Sounwave")
    - primary_artist: the album’s primary artist name
    - album_title: the album title
    - reference_urls: an array of URL(s) explicitly present in the answer that verify BOTH (a) the producer’s producer/co-producer credit on that album, and (b) the album’s 2025 release date. Include only valid URLs mentioned in the answer text. If none are present, return an empty array.

    Rules:
    1) Only one album per producer. If multiple are listed in the answer, pick the first one mentioned for that producer.
    2) The album must be released in 2025.
    3) The album must be for another primary artist (i.e., not the producer’s own solo album). Still extract what the answer presents; the evaluator will verify this constraint.
    4) Only extract URLs that are explicitly present in the answer. If a URL is missing a protocol, prepend http:// to make it valid.
    5) If any field is missing for a given producer, set it to null (or empty array for URLs).

    Output JSON schema (fill all five producers):
    {
      "dan_auerbach": {"producer": "...", "primary_artist": "...", "album_title": "...", "reference_urls": ["...", "..."]},
      "cirkut": {"producer": "...", "primary_artist": "...", "album_title": "...", "reference_urls": ["...", "..."]},
      "dijon": {"producer": "...", "primary_artist": "...", "album_title": "...", "reference_urls": ["...", "..."]},
      "blake_mills": {"producer": "...", "primary_artist": "...", "album_title": "...", "reference_urls": ["...", "..."]},
      "sounwave": {"producer": "...", "primary_artist": "...", "album_title": "...", "reference_urls": ["...", "..."]}
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def norm(s: Optional[str]) -> str:
    return (s or "").strip().casefold()


def ensure_valid_urls(urls: List[str]) -> List[str]:
    valid = []
    for u in urls:
        if not u:
            continue
        u2 = u.strip()
        if not u2:
            continue
        if not (u2.startswith("http://") or u2.startswith("https://")):
            # The extractor should already prepend, but double-safeguard here.
            u2 = "http://" + u2
        valid.append(u2)
    return valid


# --------------------------------------------------------------------------- #
# Verification routines                                                       #
# --------------------------------------------------------------------------- #
async def verify_one_producer_item(
    evaluator: Evaluator,
    parent_node,
    prefix_id: str,
    producer_name: str,
    item: ProducerAlbumItem,
) -> None:
    """
    Build the verification sub-tree for one producer item.
    """
    # Group node (non-critical to allow partial credit across producers)
    group_node = evaluator.add_parallel(
        id=f"{prefix_id}_Album",
        desc=f"One qualifying 2025 album for {producer_name} (for another primary artist)",
        parent=parent_node,
        critical=False,
    )

    # Normalize and precompute
    artist = item.primary_artist or ""
    title = item.album_title or ""
    urls = ensure_valid_urls(item.reference_urls)

    # 1) Artist provided (critical leaf as custom)
    artist_provided = bool(artist.strip())
    evaluator.add_custom_node(
        result=artist_provided,
        id=f"{prefix_id}_Artist",
        desc=f"Provide the album’s primary artist name ({producer_name} item)",
        parent=group_node,
        critical=True,
    )

    # 2) Album title provided (critical leaf as custom)
    title_provided = bool(title.strip())
    evaluator.add_custom_node(
        result=title_provided,
        id=f"{prefix_id}_Album_Title",
        desc=f"Provide the album title ({producer_name} item)",
        parent=group_node,
        critical=True,
    )

    # 3) Not producer’s own solo work (critical leaf as custom)
    # Fail if artist equals producer (case-insensitive)
    different_from_producer = (norm(artist) != norm(producer_name)) and artist_provided
    evaluator.add_custom_node(
        result=different_from_producer,
        id=f"{prefix_id}_Not_Own_Solo_Work",
        desc=f"Primary artist is not {producer_name} (i.e., not own solo work)",
        parent=group_node,
        critical=True,
    )

    # 4) Reference URL provided (critical leaf as custom)
    has_reference_url = len(urls) > 0
    ref_node = evaluator.add_custom_node(
        result=has_reference_url,
        id=f"{prefix_id}_Reference_URL",
        desc=f"Provide a reference URL from a legitimate music-industry source that verifies both the 2025 release date and {producer_name}’s producer/co-producer credit",
        parent=group_node,
        critical=True,
    )

    # Prepare shared prerequisites so verification is meaningful
    prereqs = []
    # Find the actual leaf nodes we just created to use as prerequisites
    prereqs.append(evaluator.find_node(f"{prefix_id}_Artist"))
    prereqs.append(evaluator.find_node(f"{prefix_id}_Album_Title"))
    prereqs.append(evaluator.find_node(f"{prefix_id}_Reference_URL"))
    prereqs = [p for p in prereqs if p is not None]

    # 5) Release year is 2025 (critical leaf - requires sources)
    rel_node = evaluator.add_leaf(
        id=f"{prefix_id}_Release_Year_2025",
        desc=f"Album release year is 2025 ({producer_name} item)",
        parent=group_node,
        critical=True,
    )
    release_claim = f"The album '{title}' by {artist} was released in 2025."
    await evaluator.verify(
        claim=release_claim,
        node=rel_node,
        sources=urls if has_reference_url else None,
        extra_prerequisites=prereqs,
        additional_instruction=(
            "Verify the album's release date/year on the provided page(s). The album must have a release date in 2025. "
            "Accept exact dates in 2025 (e.g., 2025-03-15). If the page is about a single or track rather than the album, "
            "or the date is not in 2025, judge as not supported."
        ),
    )

    # 6) Producer credit (critical leaf - requires sources)
    prod_node = evaluator.add_leaf(
        id=f"{prefix_id}_Production_Credit",
        desc=f"{producer_name} has a producer/co-producer credit on the album",
        parent=group_node,
        critical=True,
    )
    credit_claim = (
        f"{producer_name} is credited as a producer or co-producer on the album '{title}' by {artist}."
    )
    await evaluator.verify(
        claim=credit_claim,
        node=prod_node,
        sources=urls if has_reference_url else None,
        extra_prerequisites=prereqs,
        additional_instruction=(
            "On the provided source(s), confirm that the named person has a 'producer' or 'co-producer' credit for the full album. "
            "Accept roles such as 'producer', 'co-producer', or 'additional producer'. "
            "Do NOT accept credits that are only 'executive producer', 'mixing', 'engineering', or unrelated roles. "
            "If the page lists track-by-track production, it's acceptable as long as the person is credited for production on the album."
        ),
    )


# --------------------------------------------------------------------------- #
# Distinct primary artists check across all 5                                  #
# --------------------------------------------------------------------------- #
def add_distinct_primary_artists_check(evaluator: Evaluator, root, extracted: AllProducersExtraction) -> None:
    artists = [
        (PRODUCERS[0][1], extracted.dan_auerbach.primary_artist),
        (PRODUCERS[1][1], extracted.cirkut.primary_artist),
        (PRODUCERS[2][1], extracted.dijon.primary_artist),
        (PRODUCERS[3][1], extracted.blake_mills.primary_artist),
        (PRODUCERS[4][1], extracted.sounwave.primary_artist),
    ]
    values = [norm(a[1]) for a in artists]
    all_present = all(bool(v) for v in values)
    unique = len(set(values)) == 5 if all_present else False

    evaluator.add_custom_node(
        result=(all_present and unique),
        id="Distinct_Primary_Artists_Across_All_5",
        desc="All five selected albums have mutually different primary artists (no repeats across the five items)",
        parent=root,
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
    """
    Entry point for evaluating an answer for the Grammy 2026 Producer of the Year (Non-Classical) task.
    """
    # Initialize evaluator. Root is non-critical to allow partial credit across producers but we add a critical
    # distinct-artists check as a child to enforce that constraint strictly.
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
    extracted: AllProducersExtraction = await evaluator.extract(
        prompt=prompt_extract_producer_albums(),
        template_class=AllProducersExtraction,
        extraction_name="extracted_producer_albums",
    )

    # Optionally record ground-truth-like context for transparency (the required producer names)
    evaluator.add_ground_truth({
        "required_producers": [p[1] for p in PRODUCERS],
        "require_release_year": 2025,
        "constraint": "Album must be for another primary artist (not own solo work)",
    })

    # Build per-producer verification subtrees
    items_map: Dict[str, ProducerAlbumItem] = {
        "dan_auerbach": extracted.dan_auerbach,
        "cirkut": extracted.cirkut,
        "dijon": extracted.dijon,
        "blake_mills": extracted.blake_mills,
        "sounwave": extracted.sounwave,
    }

    for key, canonical_name, prefix in PRODUCERS:
        await verify_one_producer_item(
            evaluator=evaluator,
            parent_node=root,
            prefix_id=prefix,
            producer_name=canonical_name,
            item=items_map.get(key, ProducerAlbumItem()),
        )

    # Add cross-item distinct primary artists check (critical)
    add_distinct_primary_artists_check(evaluator, root, extracted)

    # Return the structured summary with the verification tree and scores
    return evaluator.get_summary()