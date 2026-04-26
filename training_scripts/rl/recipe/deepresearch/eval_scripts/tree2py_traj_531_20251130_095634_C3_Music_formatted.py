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
TASK_ID = "kendrick_gnx_sza_track_2026"
TASK_DESCRIPTION = """
Kendrick Lamar's album 'GNX,' released in November 2024 and nominated for Album of the Year at the 2026 Grammy Awards, includes a collaboration track featuring SZA that has garnered multiple Grammy nominations. This track was produced by a team of six producers, including both Jack Antonoff and Sounwave—two prominent producers who are both nominated for Producer of the Year, Non-Classical at the 2026 Grammy Awards.

Identify this track and provide the following verified information:

1. Track Title: The name of the track
2. Complete Production Credits: All six producers credited on the track (provide their professional/stage names as credited)
3. Mastering Engineer: The name of the mastering engineer credited for this track
4. Grammy Nominations: Confirm that this track is nominated for both Record of the Year and Song of the Year at the 2026 Grammy Awards (68th Annual Grammy Awards)
5. Producer Context:
   - Confirm that Jack Antonoff is nominated for Producer of the Year, Non-Classical at the 2026 Grammys and previously won this award in 2022, 2023, and 2024
   - Confirm that Sounwave is nominated for Producer of the Year, Non-Classical at the 2026 Grammys and that this is his first nomination in this category

All information must be supported by reference URLs from official or reputable sources.
"""

# Expected producer names to verify (as phrased in rubric)
EXPECTED_PRODUCERS = [
    "Jack Antonoff",
    "Bridgeway (Ruchaun Akers)",
    "M-Tech (Matthew Bernard)",
    "roselilah (Roshwita Larisha Bacha)",
    "Sounwave (Mark Anthony Spears)",
    "Kamasi Washington",
]


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class TrackInfoExtraction(BaseModel):
    # Core identification
    track_title: Optional[str] = None
    # Source URLs for various verifications (extract only URLs explicitly included in the answer)
    track_title_sources: List[str] = Field(default_factory=list)          # URLs that mention the track title
    album_sources: List[str] = Field(default_factory=list)                # URLs that show the track is on GNX
    sza_feature_sources: List[str] = Field(default_factory=list)          # URLs confirming SZA is featured

    # Credits
    producers: List[str] = Field(default_factory=list)                    # Producers as listed in the answer
    producers_sources: List[str] = Field(default_factory=list)            # URLs supporting producer credits
    mastering_engineer: Optional[str] = None                              # Mastering engineer as listed in the answer
    mastering_sources: List[str] = Field(default_factory=list)            # URLs supporting mastering credit

    # Grammy nominations (track-level)
    record_of_year_sources: List[str] = Field(default_factory=list)       # URLs confirming ROTY nomination
    song_of_year_sources: List[str] = Field(default_factory=list)         # URLs confirming SOTY nomination
    gnx_aoty_sources: List[str] = Field(default_factory=list)             # URLs confirming GNX AOTY nomination

    # Producer-of-the-Year context
    antonoff_poy_nom_sources: List[str] = Field(default_factory=list)     # URLs confirming 2026 POY (Non-Classical) nom for Antonoff
    antonoff_wins_sources: List[str] = Field(default_factory=list)        # URLs confirming wins in 2022, 2023, 2024
    sounwave_poy_nom_sources: List[str] = Field(default_factory=list)     # URLs confirming 2026 POY (Non-Classical) nom for Sounwave
    sounwave_first_nom_sources: List[str] = Field(default_factory=list)   # URLs confirming it is Sounwave’s first POY (Non-Classical) nomination

    # Fallback general references
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_track_info() -> str:
    return """
    From the answer, extract the following fields. Extract ONLY information explicitly present in the answer. For all URL fields, extract actual URLs mentioned in the answer (plain URLs or inside markdown links). If a field is missing, set it to null (for single value) or [] (for list).

    Required fields:
    - track_title: The exact title of the Kendrick Lamar GNX track that features SZA.
    - track_title_sources: URLs that mention this track title.
    - album_sources: URLs that show this track is from Kendrick Lamar’s album GNX (e.g., official tracklist, label/retailer/streaming service page, reputable databases).
    - sza_feature_sources: URLs confirming that SZA is a featured artist on this track.

    - producers: The list of all producers credited on the track as provided in the answer (use credited stage/professional names; include parentheses if provided).
    - producers_sources: URLs supporting the full production credits for this track (ideally a track/album credits page).
    - mastering_engineer: The mastering engineer credited for this track, as provided in the answer.
    - mastering_sources: URLs supporting the mastering engineer credit.

    - record_of_year_sources: URLs confirming the track is nominated for Record of the Year at the 2026 Grammys (68th Annual).
    - song_of_year_sources: URLs confirming the track is nominated for Song of the Year at the 2026 Grammys (68th Annual).
    - gnx_aoty_sources: URLs confirming GNX is nominated for Album of the Year at the 2026 Grammys (68th Annual).

    - antonoff_poy_nom_sources: URLs confirming Jack Antonoff is nominated for Producer of the Year, Non-Classical at the 2026 Grammys.
    - antonoff_wins_sources: URLs confirming Jack Antonoff won Producer of the Year, Non-Classical in 2022, 2023, and 2024.
    - sounwave_poy_nom_sources: URLs confirming Sounwave is nominated for Producer of the Year, Non-Classical at the 2026 Grammys.
    - sounwave_first_nom_sources: URLs confirming that 2026 is Sounwave’s first nomination for Producer of the Year, Non-Classical.

    - general_sources: Any additional general reference URLs cited in the answer that may support relevant claims.

    Notes:
    - Do NOT invent URLs. Extract only those explicitly present in the answer.
    - For names, preserve exact formatting as in the answer (including parentheses if shown).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_sources(*source_lists: Optional[List[str]]) -> Optional[List[str]]:
    """
    Merge multiple source lists into a unique ordered list. Return None if no valid URLs are found.
    """
    seen = set()
    merged: List[str] = []
    for sl in source_lists:
        if not sl:
            continue
        for url in sl:
            if not url or not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                merged.append(u)
                seen.add(u)
    return merged if merged else None


def _format_expected_producers_for_claim() -> str:
    """
    Build a human-readable semicolon-separated string of expected producers for the claim.
    """
    return "; ".join(EXPECTED_PRODUCERS)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_track_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: TrackInfoExtraction
) -> None:
    """
    Build the 'track_identification' sub-tree and perform existence check for track title.
    """
    ident_node = evaluator.add_parallel(
        id="track_identification",
        desc="Provide the track title (the specific GNX track featuring SZA).",
        parent=parent_node,
        critical=True
    )

    # Leaf: Track title provided (existence check)
    title_present = extracted.track_title is not None and extracted.track_title.strip() != ""
    evaluator.add_custom_node(
        result=title_present,
        id="track_title_provided",
        desc="Track title is explicitly stated in the answer.",
        parent=ident_node,
        critical=True
    )


async def build_track_level_verifications(
    evaluator: Evaluator,
    parent_node,
    extracted: TrackInfoExtraction
) -> None:
    """
    Build the 'track_level_verifications' sub-tree and verify all track/album/credit/nomination constraints.
    """
    track_node = evaluator.add_parallel(
        id="track_level_verifications",
        desc="Verify the identified track meets all track/album/credit/nomination constraints and provide supporting URLs.",
        parent=parent_node,
        critical=True
    )

    title = extracted.track_title or ""

    # Prepare all leaves
    leaf_track_on_gnx = evaluator.add_leaf(
        id="track_is_on_gnx_with_source",
        desc="Verify the track is from Kendrick Lamar's album 'GNX' and provide a reputable reference URL.",
        parent=track_node,
        critical=True
    )
    claim_track_on_gnx = f"The song '{title}' appears on Kendrick Lamar's album GNX."

    leaf_gnx_aoty = evaluator.add_leaf(
        id="gnx_aoty_nomination_with_source",
        desc="Verify GNX is nominated for Album of the Year at the 2026 Grammy Awards and provide a reputable reference URL.",
        parent=track_node,
        critical=True
    )
    claim_gnx_aoty = "Kendrick Lamar's album GNX is nominated for Album of the Year at the 68th Annual Grammy Awards (2026)."

    leaf_sza_feature = evaluator.add_leaf(
        id="features_sza_with_source",
        desc="Verify SZA is credited as a featured artist on the track and provide a reputable reference URL.",
        parent=track_node,
        critical=True
    )
    claim_sza_feature = f"SZA is a credited featured artist on the track '{title}'."

    leaf_producers_all_six = evaluator.add_leaf(
        id="production_credits_all_six_with_source",
        desc="Verify exactly six producers are credited on the track, and that they are: Jack Antonoff, Bridgeway (Ruchaun Akers), M-Tech (Matthew Bernard), roselilah (Roshwita Larisha Bacha), Sounwave (Mark Anthony Spears), and Kamasi Washington; provide a reputable reference URL.",
        parent=track_node,
        critical=True
    )
    producers_str = _format_expected_producers_for_claim()
    claim_producers = (
        f"The production credits for '{title}' list exactly six producers: {producers_str}."
    )

    leaf_mastering = evaluator.add_leaf(
        id="mastering_engineer_with_source",
        desc="Verify Ruairi O'Flaherty is credited as the mastering engineer for the track and provide a reputable reference URL.",
        parent=track_node,
        critical=True
    )
    claim_mastering = f"Ruairi O'Flaherty is credited as the mastering engineer for the track '{title}'."

    leaf_roty = evaluator.add_leaf(
        id="record_of_year_nomination_with_source",
        desc="Verify the track is nominated for Record of the Year at the 2026 Grammy Awards and provide a reputable reference URL.",
        parent=track_node,
        critical=True
    )
    claim_roty = f"The track '{title}' is nominated for Record of the Year at the 68th Annual Grammy Awards (2026)."

    leaf_soty = evaluator.add_leaf(
        id="song_of_year_nomination_with_source",
        desc="Verify the track is nominated for Song of the Year at the 2026 Grammy Awards and provide a reputable reference URL.",
        parent=track_node,
        critical=True
    )
    claim_soty = f"The track '{title}' is nominated for Song of the Year at the 68th Annual Grammy Awards (2026)."

    # Build claims and sources for batch verification
    claims_sources_nodes_instructions = [
        (
            claim_track_on_gnx,
            _unique_sources(extracted.album_sources, extracted.track_title_sources, extracted.general_sources),
            leaf_track_on_gnx,
            "Use reputable sources (e.g., official tracklist, label/retailer/streaming credits page, AllMusic/TIDAL credits) to confirm that the song is on the album GNX."
        ),
        (
            claim_gnx_aoty,
            _unique_sources(extracted.gnx_aoty_sources, extracted.general_sources),
            leaf_gnx_aoty,
            "Prefer official Recording Academy/Grammy sources (grammy.com) or highly reputable outlets confirming GNX's AOTY nomination for 2026 (68th Annual Grammys)."
        ),
        (
            claim_sza_feature,
            _unique_sources(extracted.sza_feature_sources, extracted.track_title_sources, extracted.album_sources, extracted.general_sources),
            leaf_sza_feature,
            "Confirm that SZA is credited as a featured artist on the track (e.g., track/album credits page, official label or streaming service pages)."
        ),
        (
            claim_producers,
            _unique_sources(extracted.producers_sources, extracted.track_title_sources, extracted.album_sources, extracted.general_sources),
            leaf_producers_all_six,
            "Verify that there are exactly six producers and that their credited names match (allowing for stage names and real names in parentheses). Accept minor punctuation/capitalization differences but ensure the set of six is exact."
        ),
        (
            claim_mastering,
            _unique_sources(extracted.mastering_sources, extracted.track_title_sources, extracted.album_sources, extracted.general_sources),
            leaf_mastering,
            "Confirm the mastering engineer credit for this specific track (prefer credits pages such as TIDAL, AllMusic, official label/engineer sites)."
        ),
        (
            claim_roty,
            _unique_sources(extracted.record_of_year_sources, extracted.general_sources),
            leaf_roty,
            "Confirm the track's nomination for Record of the Year at the 68th Annual Grammys (2026). Prefer grammy.com or highly reputable outlets."
        ),
        (
            claim_soty,
            _unique_sources(extracted.song_of_year_sources, extracted.general_sources),
            leaf_soty,
            "Confirm the track's nomination for Song of the Year at the 68th Annual Grammys (2026). Prefer grammy.com or highly reputable outlets."
        ),
    ]

    # Execute verifications (in parallel for efficiency)
    await evaluator.batch_verify(claims_sources_nodes_instructions)


async def build_producer_context_verifications(
    evaluator: Evaluator,
    parent_node,
    extracted: TrackInfoExtraction
) -> None:
    """
    Build the 'producer_of_year_context_verifications' sub-tree and verify Producer of the Year, Non-Classical claims.
    """
    ctx_node = evaluator.add_parallel(
        id="producer_of_year_context_verifications",
        desc="Verify Producer of the Year, Non-Classical nomination context for Jack Antonoff and Sounwave with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # Leaves
    leaf_antonoff_nom = evaluator.add_leaf(
        id="antonoff_poy_nomination_with_source",
        desc="Verify Jack Antonoff is nominated for Producer of the Year, Non-Classical at the 2026 Grammys and provide a reputable reference URL.",
        parent=ctx_node,
        critical=True
    )
    claim_antonoff_nom = "Jack Antonoff is nominated for Producer of the Year, Non-Classical at the 68th Annual Grammy Awards (2026)."

    leaf_antonoff_wins = evaluator.add_leaf(
        id="antonoff_three_consecutive_wins_with_source",
        desc="Verify Jack Antonoff previously won Producer of the Year, Non-Classical in 2022, 2023, and 2024 (three consecutive years) and provide a reputable reference URL.",
        parent=ctx_node,
        critical=True
    )
    claim_antonoff_wins = "Jack Antonoff won Producer of the Year, Non-Classical in 2022, 2023, and 2024."

    leaf_sounwave_nom = evaluator.add_leaf(
        id="sounwave_poy_nomination_with_source",
        desc="Verify Sounwave is nominated for Producer of the Year, Non-Classical at the 2026 Grammys and provide a reputable reference URL.",
        parent=ctx_node,
        critical=True
    )
    claim_sounwave_nom = "Sounwave (Mark Anthony Spears) is nominated for Producer of the Year, Non-Classical at the 68th Annual Grammy Awards (2026)."

    leaf_sounwave_first = evaluator.add_leaf(
        id="sounwave_first_nomination_with_source",
        desc="Verify this is Sounwave's first nomination for Producer of the Year, Non-Classical and provide a reputable reference URL.",
        parent=ctx_node,
        critical=True
    )
    claim_sounwave_first = "The 2026 nomination is Sounwave's first-ever nomination for Producer of the Year, Non-Classical."

    # Prepare for batch verification
    claims_sources_nodes_instructions = [
        (
            claim_antonoff_nom,
            _unique_sources(extracted.antonoff_poy_nom_sources, extracted.general_sources),
            leaf_antonoff_nom,
            "Prefer grammy.com or other reputable outlets listing 2026 Producer of the Year, Non-Classical nominees."
        ),
        (
            claim_antonoff_wins,
            _unique_sources(extracted.antonoff_wins_sources, extracted.general_sources),
            leaf_antonoff_wins,
            "Verify Antonoff's wins for Producer of the Year, Non-Classical in 2022, 2023, and 2024 (accept reputable outlets or grammy.com)."
        ),
        (
            claim_sounwave_nom,
            _unique_sources(extracted.sounwave_poy_nom_sources, extracted.general_sources),
            leaf_sounwave_nom,
            "Prefer grammy.com or reputable outlets confirming Sounwave's 2026 nomination for Producer of the Year, Non-Classical."
        ),
        (
            claim_sounwave_first,
            _unique_sources(extracted.sounwave_first_nom_sources, extracted.general_sources),
            leaf_sounwave_first,
            "Confirm that Sounwave had no prior nominations in this category before 2026; check historical nominee/winner pages on grammy.com or reputable outlets."
        ),
    ]

    # Execute verifications (in parallel)
    await evaluator.batch_verify(claims_sources_nodes_instructions)


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
    Evaluate an answer for the Kendrick Lamar GNX + SZA track verification task using the Mind2Web2 framework.
    """
    # Initialize evaluator with sequential root to reflect ordered gating
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_track_info(),
        template_class=TrackInfoExtraction,
        extraction_name="track_info"
    )

    # Record expected producers for transparency
    evaluator.add_custom_info(
        info={"expected_producers": EXPECTED_PRODUCERS},
        info_type="expectations",
        info_name="expected_production_credits"
    )

    # Build tree and run verifications following rubric order
    # 1) Track identification (title presence)
    await build_track_identification(evaluator, root, extracted_info)

    # 2) Track-level verifications (will be skipped if track title not provided due to sequential gating)
    await build_track_level_verifications(evaluator, root, extracted_info)

    # 3) Producer-of-the-Year context verifications (also gated by previous steps due to sequential root)
    await build_producer_context_verifications(evaluator, root, extracted_info)

    # Return evaluation summary
    return evaluator.get_summary()