import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "riaa_album_dec_2025"
TASK_DESCRIPTION = (
    "Identify one album that received an RIAA Gold or Platinum certification during December 2025. "
    "Provide the album title and artist, producer(s) with a verifiable source, the record label as listed in the RIAA certification, "
    "and the exact certification date (month, day, year) from the RIAA database."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AlbumExtraction(BaseModel):
    album_title: Optional[str] = None
    artist_name: Optional[str] = None

    riaa_urls: List[str] = Field(default_factory=list)  # URLs to the specific RIAA Gold & Platinum entries
    certification_date: Optional[str] = None           # Exact date string as stated in the answer, e.g., "December 3, 2025"
    certification_level: Optional[str] = None          # e.g., "Gold", "Platinum", "Multi-Platinum"
    certification_format: Optional[str] = None         # e.g., "Album", "Single"
    record_label: Optional[str] = None                 # Label shown in RIAA entry

    producer_names: List[str] = Field(default_factory=list)
    producer_sources: List[str] = Field(default_factory=list)  # URLs like Wikipedia, Discogs, AllMusic, official credits


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_album_info() -> str:
    return (
        "Extract the following fields from the answer for one album certified by the RIAA in December 2025:\n"
        "- album_title: The album title\n"
        "- artist_name: The primary artist name\n"
        "- riaa_urls: A list of URLs pointing to the album’s entry in the RIAA Gold & Platinum database (exact URLs). "
        "If none are provided, return an empty list.\n"
        "- certification_date: The exact certification date as written in the answer (month, day, year). If missing, return null.\n"
        "- certification_level: The award level (e.g., Gold, Platinum, Multi-Platinum). If missing, return null.\n"
        "- certification_format: The format stated in the RIAA entry (e.g., Album, Single). If missing, return null.\n"
        "- record_label: The record label listed in the RIAA entry. If missing, return null.\n"
        "- producer_names: A list of the album’s producer names mentioned in the answer. If none, return an empty list.\n"
        "- producer_sources: A list of URLs that verify the producer credits (e.g., Wikipedia, Discogs, AllMusic, Spotify credits, official liner notes). "
        "If none are provided, return an empty list.\n\n"
        "Rules:\n"
        "1) Only extract information explicitly present in the answer.\n"
        "2) For any URL fields, include full URLs. If missing protocol, prepend http://.\n"
        "3) If a field is not mentioned, use null or an empty list as appropriate.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in urls:
        if u and (u not in seen):
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_album_eligibility(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: AlbumExtraction,
) -> VerificationNode:
    """
    Build and verify the 'Album_Eligibility' subtree.
    This subtree checks:
    - Title and artist correspond to the RIAA page
    - Certification date is within Dec 1–31, 2025 (inclusive)
    - Certification level is Gold or any Platinum-tier
    - Certification format is ALBUM (not single, not EP)
    Also adds an existence gate for RIAA source URLs.
    """
    node = evaluator.add_parallel(
        id="Album_Eligibility",
        desc="The identified release satisfies the RIAA certification eligibility constraints (date window, award type, and album format) as shown in the RIAA Gold & Platinum database.",
        parent=parent,
        critical=True,
    )

    # Gate: RIAA source URL(s) must be provided to verify certification-related facts
    riaa_urls = _safe_sources(data.riaa_urls)
    riaa_source_gate = evaluator.add_custom_node(
        result=len(riaa_urls) > 0,
        id="RIAA_Source_Provided",
        desc="At least one RIAA Gold & Platinum database URL is provided to verify certification details.",
        parent=node,
        critical=True,
    )

    # Leaf: Album title and artist verification using RIAA page(s)
    title_artist_leaf = evaluator.add_leaf(
        id="Album_Title_and_Artist",
        desc="Provides the album title and the artist name for the identified certified release.",
        parent=node,
        critical=True,
    )
    claim_title_artist = (
        f"On the provided RIAA Gold & Platinum database page(s), the certified release is an album titled "
        f"'{data.album_title or ''}' by '{data.artist_name or ''}'."
    )
    # Leaf: Certification date range verification on RIAA page(s)
    date_range_leaf = evaluator.add_leaf(
        id="Certification_Date_Range",
        desc="RIAA certification date is between December 1, 2025 and December 31, 2025 (inclusive), as verified in the RIAA Gold & Platinum database.",
        parent=node,
        critical=True,
    )
    claim_date_range = (
        "The certification date shown on the RIAA Gold & Platinum database entry falls between December 1, 2025 and December 31, 2025 (inclusive)."
    )

    # Leaf: Certification level (Gold or Platinum-tier)
    level_leaf = evaluator.add_leaf(
        id="Certification_Level",
        desc="RIAA certification award level is Gold or a Platinum-tier certification (as displayed in the RIAA Gold & Platinum database).",
        parent=node,
        critical=True,
    )
    claimed_level = data.certification_level or ""
    claim_level = (
        f"The certification level on the RIAA entry is Gold or a Platinum-tier (including Multi-Platinum); the answer lists '{claimed_level}'."
    )

    # Leaf: Album format is ALBUM
    format_leaf = evaluator.add_leaf(
        id="Album_Format",
        desc="RIAA certification format is designated as 'ALBUM' (not single, not EP) in the RIAA database entry.",
        parent=node,
        critical=True,
    )
    claimed_format = data.certification_format or ""
    claim_format = (
        f"The RIAA entry indicates the format is 'Album' (not Single or EP); the answer lists '{claimed_format}'."
    )

    # Verify all four leaves (critical siblings) with the RIAA source gate as prerequisite
    await evaluator.batch_verify(
        [
            (
                claim_title_artist,
                riaa_urls,
                title_artist_leaf,
                "Focus on the 'Title' and 'Artist' fields on the RIAA page. Allow minor punctuation or capitalization differences.",
            ),
            (
                claim_date_range,
                riaa_urls,
                date_range_leaf,
                "Use the 'Certification Date' field on the RIAA page to decide if it falls within December 2025. Inclusive of Dec 1–Dec 31.",
            ),
            (
                claim_level,
                riaa_urls,
                level_leaf,
                "Consider any Platinum-tier (e.g., Platinum, Multi-Platinum) as acceptable. The page must clearly indicate Gold or a Platinum-tier.",
            ),
            (
                claim_format,
                riaa_urls,
                format_leaf,
                "Check the 'Format' or equivalent field. It must explicitly indicate 'Album' and not 'Single' or 'EP'.",
            ),
        ]
    )

    return node


async def build_album_metadata(
    evaluator: Evaluator,
    parent: VerificationNode,
    data: AlbumExtraction,
) -> VerificationNode:
    """
    Build and verify the 'Album_Metadata' subtree.
    This subtree checks:
    - Producers with verifiable sources (non-RIAA sources acceptable)
    - Record label matches the RIAA entry
    - Exact certification date matches the RIAA entry
    """
    node = evaluator.add_parallel(
        id="Album_Metadata",
        desc="Provides the required album-related metadata and citations/verification sources as requested.",
        parent=parent,
        critical=True,
    )

    # Producer information verification (uses producer_sources)
    producer_leaf = evaluator.add_leaf(
        id="Producer_Information",
        desc="Lists the producer(s) for the album and provides a verifiable reference source for the producer credits (e.g., Wikipedia, Discogs, AllMusic, Spotify credits, or official liner notes).",
        parent=node,
        critical=True,
    )
    producers_list = data.producer_names or []
    producer_sources = _safe_sources(data.producer_sources)
    claim_producers = (
        f"The album '{data.album_title or ''}' by '{data.artist_name or ''}' credits the following producers: {producers_list}."
    )

    # Record label verification against RIAA page(s)
    label_leaf = evaluator.add_leaf(
        id="Record_Label",
        desc="Provides the record label as listed in the RIAA certification entry.",
        parent=node,
        critical=True,
    )
    riaa_urls = _safe_sources(data.riaa_urls)
    claim_label = (
        f"The record label shown in the RIAA Gold & Platinum entry is '{data.record_label or ''}'."
    )

    # Exact certification date verification against RIAA page(s)
    exact_date_leaf = evaluator.add_leaf(
        id="Exact_Certification_Date",
        desc="Provides the exact RIAA certification date (month, day, year) from the RIAA database.",
        parent=node,
        critical=True,
    )
    claim_exact_date = (
        f"The exact certification date on the RIAA entry is '{data.certification_date or ''}'."
    )

    # Verify producer information via producer_sources
    await evaluator.verify(
        claim=claim_producers,
        node=producer_leaf,
        sources=producer_sources,
        additional_instruction=(
            "Verify producer credits using the provided sources (e.g., Wikipedia, Discogs, AllMusic, Spotify credits, "
            "official liner notes). The listed names should appear as producers for this album on the source page(s). "
            "Allow minor name variants or capitalization differences."
        ),
    )

    # Verify label and exact date via RIAA page(s)
    await evaluator.batch_verify(
        [
            (
                claim_label,
                riaa_urls,
                label_leaf,
                "Check the 'Label' or equivalent field on the RIAA entry and confirm it matches the stated label.",
            ),
            (
                claim_exact_date,
                riaa_urls,
                exact_date_leaf,
                "Check the 'Certification Date' on the RIAA entry and confirm it matches the exact stated date (month, day, year). Allow standard formatting variants like abbreviated month names.",
            ),
        ]
    )

    return node


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
    Evaluate an answer for the RIAA album certification task (December 2025).
    Returns the evaluation summary with the verification tree and final score.
    """
    # Initialize evaluator with sequential root to enforce gating between major phases
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract album info from the answer
    extracted: AlbumExtraction = await evaluator.extract(
        prompt=prompt_extract_album_info(),
        template_class=AlbumExtraction,
        extraction_name="album_extraction",
    )

    # Add helpful context info to summary
    evaluator.add_custom_info(
        info={
            "required_window": {"start": "2025-12-01", "end": "2025-12-31"},
            "acceptable_levels": ["Gold", "Platinum", "Multi-Platinum (any tier)"],
            "required_format": "Album",
        },
        info_type="constraints",
        info_name="eligibility_constraints",
    )

    # Build and verify eligibility subtree
    eligibility_node = await build_album_eligibility(evaluator, root, extracted)

    # Build and verify metadata subtree (will be auto-skipped if eligibility fails due to sequential root)
    await build_album_metadata(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()