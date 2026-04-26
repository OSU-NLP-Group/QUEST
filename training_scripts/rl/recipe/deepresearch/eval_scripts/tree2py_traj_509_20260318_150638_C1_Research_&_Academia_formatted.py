import asyncio
import logging
from typing import Any, List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eclipse_2026_entirety_regions"
TASK_DESCRIPTION = """
Which regions of the world can observe the total lunar eclipse on March 3, 2026, in its entirety?
"""

GROUND_TRUTH_REGIONS = [
    "Western North America",
    "Oceania",
    "Asia",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RegionEvidence(BaseModel):
    """A region mentioned in the answer as having entire/complete visibility, with any cited URLs."""
    name: Optional[str] = None
    mentions_entirety: Optional[bool] = None
    support_urls: List[str] = Field(default_factory=list)


class RegionsExtraction(BaseModel):
    """Extraction of regions and any (region-level or global) cited URLs from the answer."""
    regions: List[RegionEvidence] = Field(default_factory=list)
    global_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_regions() -> str:
    return """
    From the answer, extract the regions that are claimed to be able to view the total lunar eclipse on March 3, 2026 IN ITS ENTIRETY (i.e., from start to finish, whole/entire eclipse visible, entire event visible, all phases visible). 
    Rules:
    - Only include a region in the 'regions' list if the answer implies the eclipse is fully/entirely visible there using phrases such as: "entire eclipse visible", "entire event visible", "from start to finish", "whole eclipse", "all phases visible", "visible in its entirety".
    - For each such region, collect any URLs explicitly cited in the answer that support that specific region’s claim. Place those URLs in 'support_urls'.
    - Also collect any URLs cited generally as sources for visibility details (not tied to a particular region) into 'global_urls'.

    Return JSON with:
    - regions: array of objects { name, mentions_entirety (boolean), support_urls (array of URLs) }
    - global_urls: array of URLs cited generally in the answer for this eclipse’s visibility

    Notes:
    - Keep region names as they appear in the answer (e.g., "Western North America", "Oceania", "Asia", "US West Coast", "western Canada", etc.).
    - The URL fields must be valid URLs that appear in the answer text. Do not invent any.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_region_aliases() -> Dict[str, List[str]]:
    """Canonical regions and a set of relaxed aliases that commonly appear."""
    return {
        "western_north_america": [
            "western north america",
            "western portion of north america",
            "western part of north america",
            "us west coast",
            "west coast of north america",
            "western united states",
            "western usa",
            "western u.s.",
            "western canada",
            "alaska",
            "british columbia",
            "pacific northwest",
        ],
        "oceania": [
            "oceania",
            "australia",
            "new zealand",
            "papua new guinea",
            "melanesia",
            "polynesia",
            "micronesia",
            "pacific islands",
        ],
        "asia": [
            "asia",
            "asian continent",
            "east asia",
            "southeast asia",
            "south asia",
            "central asia",
            "west asia",
            "middle east",
        ],
    }


def canonical_name_for(region_id: str) -> str:
    mapping = {
        "western_north_america": "Western North America",
        "oceania": "Oceania",
        "asia": "Asia",
    }
    return mapping.get(region_id, region_id)


def normalize_and_filter_urls(urls: List[str]) -> List[str]:
    """Deduplicate and keep only plausible HTTP(S) URLs."""
    out = []
    seen = set()
    for u in urls or []:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        # Accept http/https; if scheme missing but looks like a domain, prepend http://
        if not s.lower().startswith(("http://", "https://")):
            # Heuristic: add scheme if it looks like a domain
            if "." in s and " " not in s:
                s = "http://" + s
            else:
                continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def collect_region_sources(extraction: RegionsExtraction, region_id: str) -> List[str]:
    """Collect URLs cited for a particular region; fall back to global URLs if per‑region is empty."""
    aliases = [a.lower() for a in get_region_aliases().get(region_id, [])]
    per_region: List[str] = []
    for r in extraction.regions:
        if not r or not r.name:
            continue
        name_l = r.name.lower()
        # relaxed containment match either way
        if any(a in name_l or name_l in a for a in aliases):
            per_region.extend(r.support_urls or [])
    per_region = normalize_and_filter_urls(per_region)
    if per_region:
        return per_region

    # Fallback to any general sources from the answer
    return normalize_and_filter_urls(extraction.global_urls or [])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_region_entirety(
    evaluator: Evaluator,
    parent_node,
    region_id: str,
    extraction: RegionsExtraction,
) -> None:
    """
    Build a sequential verification for a region:
      1) The answer claims the region can see the eclipse in its entirety (simple check against the answer text).
      2) The answer provides at least one source URL for this claim (custom existence check).
      3) The cited sources support the claim (URL‑grounded verification).
    """
    canonical = canonical_name_for(region_id)
    region_node = evaluator.add_sequential(
        id=region_id,
        desc=f"{canonical} is identified as a region with complete eclipse visibility",
        parent=parent_node,
        critical=False,  # allow partial credit across regions
    )

    # 1) Claim appears in the answer text
    claim_leaf = evaluator.add_leaf(
        id=f"{region_id}_claimed",
        desc=f"Answer identifies {canonical} as a region with complete eclipse visibility",
        parent=region_node,
        critical=True,
    )
    # Phrase claim generously but focused; let LLM allow reasonable synonyms
    existence_claim = (
        f"The answer states or clearly implies that {canonical} can view the total lunar eclipse on March 3, 2026 "
        f"in its entirety (e.g., 'entire eclipse visible', 'entire event visible', 'from start to finish', "
        f"'whole eclipse visible', or 'all phases visible')."
    )
    await evaluator.verify(
        claim=existence_claim,
        node=claim_leaf,
        additional_instruction=(
            "Judge only based on the provided answer text. Allow regional paraphrases and subregion groupings "
            "(e.g., 'western U.S./Canada' for Western North America; Australia/New Zealand collectively for Oceania; "
            "East/Southeast Asia for Asia). Minor wording differences are acceptable as long as the meaning is that "
            "the entire eclipse is visible in that region."
        ),
    )

    # 2) Sources are provided (existence check)
    sources = collect_region_sources(extraction, region_id)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id=f"{region_id}_sources_provided",
        desc=f"Source URLs are provided in the answer for the {canonical} claim",
        parent=region_node,
        critical=True,
    )

    # 3) Sources support the claim
    support_leaf = evaluator.add_leaf(
        id=f"{region_id}_supported",
        desc=f"Cited sources support that {canonical} can see the March 3, 2026 total lunar eclipse in its entirety",
        parent=region_node,
        critical=True,
    )
    support_claim = (
        f"{canonical} can view the total lunar eclipse on March 3, 2026 in its entirety (from start to finish)."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_leaf,
        sources=sources,
        additional_instruction=(
            "Use the provided webpage(s) to confirm the 'entire eclipse visible' (or equivalent) status for this region. "
            "Acceptance criteria include explicit statements or visibility maps indicating 'entire eclipse visible', "
            "'entire event visible', 'whole eclipse visible', or that the Moon is above the horizon for the full duration "
            "of the eclipse in this region."
        ),
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
    Evaluate an answer for the regions that can observe the March 3, 2026 total lunar eclipse in its entirety.
    """
    # Initialize evaluator (root kept non‑critical to allow partial credit across regions)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identifies all regions that can view the March 3, 2026 total lunar eclipse in its entirety",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_regions(),
        template_class=RegionsExtraction,
        extraction_name="regions_entire_eclipse_extraction",
    )

    # Record ground truth (for transparency; not used for hard matching)
    evaluator.add_ground_truth(
        {
            "expected_regions_general_labels": GROUND_TRUTH_REGIONS,
            "task": "Identify regions with entire (start-to-finish) visibility for the March 3, 2026 total lunar eclipse",
        }
    )

    # Build per‑region verifications
    await verify_region_entirety(evaluator, root, "western_north_america", extraction)
    await verify_region_entirety(evaluator, root, "oceania", extraction)
    await verify_region_entirety(evaluator, root, "asia", extraction)

    # Return structured summary
    return evaluator.get_summary()