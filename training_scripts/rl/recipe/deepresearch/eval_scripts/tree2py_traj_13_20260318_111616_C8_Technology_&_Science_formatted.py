import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "may2024_geomagnetic_impacts"
TASK_DESCRIPTION = (
    "During the geomagnetic storm event of May 10-11, 2024 (the strongest in 20 years), "
    "identify four distinct infrastructure sectors that experienced documented impacts consistent with "
    "at least G3 (Strong) level effects according to NOAA's Space Weather Scale. For each sector, provide: "
    "(1) a description of the specific impact observed, (2) the corresponding NOAA geomagnetic storm scale level "
    "(G3, G4, or G5) that describes this type of impact, and (3) a reference URL from a credible source documenting "
    "the impact during this specific storm event."
)

STORM_WINDOW_TEXT = "May 10–11, 2024"
NOAA_SCALE_SOURCES = [
    # NOAA Space Weather Prediction Center official G-scale pages
    "https://www.swpc.noaa.gov/noaa-scales/geomagnetic-storms",
    "https://www.swpc.noaa.gov/phenomena/geomagnetic-storms",
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SectorItem(BaseModel):
    sector_name: Optional[str] = None
    impact_description: Optional[str] = None
    noaa_scale_level: Optional[str] = None  # Expect "G3", "G4", or "G5"
    reference_urls: List[str] = Field(default_factory=list)


class SectorsExtraction(BaseModel):
    sectors: List[SectorItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_sectors() -> str:
    return """
    Extract up to four distinct infrastructure sectors from the answer that reportedly experienced impacts during the May 10–11, 2024 geomagnetic storm event.
    For each sector, extract the following fields:
    - sector_name: The infrastructure sector name (e.g., power grid, satellites, aviation, GPS/GNSS, railways, pipelines, etc.). Use a concise sector term.
    - impact_description: A clear, specific impact that occurred in this sector during the May 10–11, 2024 storm (e.g., voltage control issues, GPS degradation, aviation disruptions, satellite anomalies).
    - noaa_scale_level: The NOAA geomagnetic storm scale level label corresponding to this type of impact. Must be exactly one of "G3", "G4", or "G5". If the answer mentions descriptors (e.g., "Strong", "Severe", "Extreme"), convert them to the exact code ("G3", "G4", "G5").
    - reference_urls: A list of credible URL(s) directly documenting that this specific impact occurred during the May 10–11, 2024 event. Only include URLs explicitly present in the answer text; do not invent or infer URLs.

    Notes:
    - Only include sectors where an impact during May 10–11, 2024 is actually documented by a URL in the answer.
    - Prefer official or reputable sources (e.g., NOAA, government agencies, major operators, mainstream media, recognized scientific/space-weather outlets).
    - If any required field for a sector is missing, still include the sector object but set missing fields to null (and use an empty array for reference_urls if none are provided).
    - Return a JSON object with a top-level array field named "sectors".
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _normalize_scale_label(label: Optional[str]) -> str:
    if not label:
        return ""
    return label.strip().upper()


# --------------------------------------------------------------------------- #
# Sector verification                                                         #
# --------------------------------------------------------------------------- #
async def verify_sector(
    evaluator: Evaluator,
    parent_node,
    sector: SectorItem,
    sector_index: int,
) -> None:
    """
    Build verification sub-tree for a single sector:
    - Required fields present (critical gating)
    - Impact aligns with NOAA scale definitions (critical)
    - NOAA level is a valid G3/G4/G5 label (critical)
    - Reference URL documents the specific impact during May 10–11, 2024 (critical)
    """

    # Create a parallel node for this sector
    sector_node = evaluator.add_parallel(
        id=f"sector_{sector_index+1}",
        desc=f"Documented infrastructure sector #{sector_index+1} experiencing G3-or-higher impacts",
        parent=parent_node,
        critical=False,
    )

    # Prepare normalized values
    sector_name = (sector.sector_name or "").strip()
    impact_desc = (sector.impact_description or "").strip()
    scale_label = _normalize_scale_label(sector.noaa_scale_level)
    urls = sector.reference_urls or []

    # Add a critical "required fields" gating node (existence and basic validity)
    required_ok = (
        bool(sector_name)
        and bool(impact_desc)
        and scale_label in {"G3", "G4", "G5"}
        and isinstance(urls, list)
        and len(urls) > 0
    )
    evaluator.add_custom_node(
        result=required_ok,
        id=f"sector_{sector_index+1}_required_fields",
        desc=f"Sector #{sector_index+1}: required fields present (sector name, specific impact, G3/G4/G5 label, and at least one reference URL)",
        parent=sector_node,
        critical=True,
    )

    # 1) Impact aligns with NOAA scale definitions (critical)
    impact_align_node = evaluator.add_leaf(
        id=f"sector_{sector_index+1}_impact_alignment",
        desc="Specific impact aligns with NOAA G3/G4/G5 scale definitions",
        parent=sector_node,
        critical=True,
    )
    impact_claim = (
        f"The described impact for this sector is consistent with NOAA geomagnetic storm level {scale_label}. "
        f"Impact provided: '{impact_desc}'."
    )
    await evaluator.verify(
        claim=impact_claim,
        node=impact_align_node,
        sources=NOAA_SCALE_SOURCES,  # Verify alignment against NOAA's official G-scale descriptions
        additional_instruction=(
            "Judge whether the stated impact is a typical or expected effect at the given NOAA G-scale level. "
            "Use the official NOAA Space Weather Scales pages provided as evidence. "
            "Allow reasonable synonyms or paraphrases of impacts. If the impact is not clearly matched to the given level, mark as not supported."
        ),
    )

    # 2) NOAA scale level is a valid label G3/G4/G5 (critical)
    level_valid_node = evaluator.add_leaf(
        id=f"sector_{sector_index+1}_noaa_level_valid",
        desc="NOAA geomagnetic storm scale level (G3/G4/G5) is valid",
        parent=sector_node,
        critical=True,
    )
    level_claim = f"The NOAA geomagnetic storm scale level value is valid (one of G3, G4, or G5): '{scale_label}'."
    await evaluator.verify(
        claim=level_claim,
        node=level_valid_node,
        additional_instruction=(
            "This is a simple logical check about the label's validity only (G3/G4/G5). "
            "Do not re-evaluate the impact-event mapping here."
        ),
    )

    # 3) Reference URL documents the specific impact during the May 10–11, 2024 storm (critical)
    ref_node = evaluator.add_leaf(
        id=f"sector_{sector_index+1}_reference_url",
        desc="Valid URL documents the specific impact during May 10–11, 2024 storm",
        parent=sector_node,
        critical=True,
    )
    ref_claim = (
        f"During the geomagnetic storm of {STORM_WINDOW_TEXT}, the '{sector_name}' sector experienced: {impact_desc}. "
        "This source documents that this specific impact occurred during that event."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=urls,  # Use the provided URLs from the answer
        additional_instruction=(
            "Verify that at least one provided URL is a credible source explicitly documenting the stated impact "
            f"and that it occurred during {STORM_WINDOW_TEXT}. Accept timezone variations (UTC/local) as long as the event "
            "clearly refers to the May 10–11, 2024 storm. The page may be updated later, but it must clearly describe the impact "
            "as having occurred during that event. If URLs are irrelevant, inaccessible, or do not mention the specific event timeframe or impact, mark as not supported."
        ),
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the May 2024 geomagnetic storm infrastructure impacts task.
    """

    # Initialize evaluator (root is non-critical, parallel aggregation)
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

    # Extract sector information
    extracted = await evaluator.extract(
        prompt=prompt_extract_sectors(),
        template_class=SectorsExtraction,
        extraction_name="sectors_extraction",
    )

    # Record useful GT/context info
    evaluator.add_ground_truth({
        "required_min_scale": "G3",
        "storm_window": STORM_WINDOW_TEXT,
        "noaa_scale_sources": NOAA_SCALE_SOURCES,
        "expected_num_sectors": 4
    })

    # Prepare exactly four sectors (pad with empty if fewer; truncate if more)
    sectors: List[SectorItem] = list(extracted.sectors or [])
    sectors = sectors[:4]
    while len(sectors) < 4:
        sectors.append(SectorItem())

    # Build sector verification subtrees
    tasks = []
    for i, sec in enumerate(sectors):
        tasks.append(verify_sector(evaluator, evaluator.root, sec, i))
    await asyncio.gather(*tasks)

    # Return evaluator summary
    return evaluator.get_summary()