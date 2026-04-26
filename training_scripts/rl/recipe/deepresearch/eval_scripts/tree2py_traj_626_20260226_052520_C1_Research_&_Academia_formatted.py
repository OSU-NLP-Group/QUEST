import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "march_2026_eclipse_info"
TASK_DESCRIPTION = (
    "According to NASA's official information about the March 3, 2026 total lunar eclipse, "
    "what is the duration of the total phase (totality), and in which constellation will the Moon be positioned during the eclipse?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EclipseInfo(BaseModel):
    """
    Structured extraction of the key facts from the agent's answer for the March 3, 2026 lunar eclipse.
    """
    totality_duration: Optional[str] = None  # e.g., "58 minutes", "about one hour"
    moon_constellation: Optional[str] = None  # e.g., "Leo"
    sources: List[str] = Field(default_factory=list)  # URLs explicitly cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_info() -> str:
    return """
    Extract the key facts the answer provides about the March 3, 2026 total lunar eclipse.
    Return the following fields:
    1) totality_duration: The duration of the total phase (totality) exactly as stated in the answer.
       Examples that should be captured as-is include "58 minutes", "about one hour", "~1 hour".
       If not provided, return null.
    2) moon_constellation: The constellation in which the Moon is positioned during the eclipse exactly as stated in the answer.
       Example: "Leo". If not provided, return null.
    3) sources: Extract all URLs explicitly mentioned in the answer text that are presented as sources for these facts.
       - Include only valid URLs explicitly present in the answer (plain URLs or markdown links).
       - Do not invent any URLs.
       - If none are provided, return an empty list.

    IMPORTANT:
    - Do not add, omit, or infer any information. Only extract what the answer explicitly states.
    - Keep durations and constellation names exactly as in the answer (including qualifiers like "approximately").
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def filter_nasa_urls(urls: List[str]) -> List[str]:
    """
    Filter only NASA official URLs from a list of URLs.
    Accept common NASA domains including gsfc.nasa.gov (NASA Goddard) and general nasa.gov subdomains.
    """
    nasa_domains = ("nasa.gov", "gsfc.nasa.gov", "eclipse.gsfc.nasa.gov", "science.nasa.gov")
    filtered = []
    for u in urls:
        try:
            lu = u.lower()
        except Exception:
            lu = str(u).lower()
        if any(dom in lu for dom in nasa_domains):
            filtered.append(u)
    return filtered


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_duration_of_totality(
    evaluator: Evaluator,
    parent_node,
    info: EclipseInfo,
) -> None:
    """
    Build and run the verification steps for the duration of totality.
    Gate verification on existence of both a claimed duration and at least one NASA URL.
    """
    # Organize checks sequentially: existence -> source-backed verification
    seq_node = evaluator.add_sequential(
        id="Duration_of_Totality_main",
        desc="Pipeline: Verify duration of totality against NASA official sources",
        parent=parent_node,
        critical=False
    )

    nasa_urls = filter_nasa_urls(info.sources)
    has_duration = bool(info.totality_duration and info.totality_duration.strip())
    has_nasa_source = len(nasa_urls) > 0

    # Existence and provenance gate (critical)
    evaluator.add_custom_node(
        result=(has_duration and has_nasa_source),
        id="Duration_of_Totality_presence",
        desc="Answer includes a totality duration and cites at least one NASA official URL",
        parent=seq_node,
        critical=True
    )

    # Leaf verification: Does NASA support the stated duration?
    duration_leaf = evaluator.add_leaf(
        id="Duration_of_Totality",
        desc="Correctly states that the duration of the total phase is 58 minutes (or approximately one hour)",
        parent=seq_node,
        critical=True
    )

    claim = (
        f"The duration of the total phase (totality) of the March 3, 2026 total lunar eclipse is {info.totality_duration}."
    )
    await evaluator.verify(
        claim=claim,
        node=duration_leaf,
        sources=nasa_urls,
        additional_instruction=(
            "Use the NASA official eclipse page(s) for March 3, 2026. "
            "Consider '58 minutes' and phrases like 'about one hour' as equivalent if the NASA page shows a totality duration close to 58 minutes. "
            "Minor rounding or formatting differences (e.g., '58m', '0:58') should be accepted."
        ),
    )


async def verify_constellation_location(
    evaluator: Evaluator,
    parent_node,
    info: EclipseInfo,
) -> None:
    """
    Build and run the verification steps for the Moon's constellation during the eclipse.
    Gate verification on existence of a constellation claim and at least one NASA URL.
    """
    # Organize checks sequentially: existence -> source-backed verification
    seq_node = evaluator.add_sequential(
        id="Constellation_Location_main",
        desc="Pipeline: Verify the Moon's constellation against NASA official sources",
        parent=parent_node,
        critical=False
    )

    nasa_urls = filter_nasa_urls(info.sources)
    has_constellation = bool(info.moon_constellation and info.moon_constellation.strip())
    has_nasa_source = len(nasa_urls) > 0

    # Existence and provenance gate (critical)
    evaluator.add_custom_node(
        result=(has_constellation and has_nasa_source),
        id="Constellation_Location_presence",
        desc="Answer includes a constellation for the Moon and cites at least one NASA official URL",
        parent=seq_node,
        critical=True
    )

    # Leaf verification: Does NASA support the stated constellation?
    constellation_leaf = evaluator.add_leaf(
        id="Constellation_Location",
        desc="Correctly identifies that the Moon will be positioned in the constellation Leo during the eclipse",
        parent=seq_node,
        critical=True
    )

    claim = (
        f"During the March 3, 2026 total lunar eclipse, the Moon will be positioned in the constellation {info.moon_constellation}."
    )
    await evaluator.verify(
        claim=claim,
        node=constellation_leaf,
        sources=nasa_urls,
        additional_instruction=(
            "Verify on NASA's official eclipse resources for March 3, 2026 that the Moon is in the stated constellation. "
            "Allow minor naming or casing variations (e.g., 'leo' vs 'Leo'). "
            "If NASA explicitly lists 'Leo' for the Moon's position, that should support the claim."
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
    Evaluate an answer for the NASA March 3, 2026 total lunar eclipse key information task.

    Returns:
        A structured summary dictionary containing the verification tree, scores, and recorded extractions.
    """
    # Initialize evaluator with a parallel root (two independent key facts)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Provides accurate key information about the March 3, 2026 total lunar eclipse as requested",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract key facts from the answer
    info = await evaluator.extract(
        prompt=prompt_extract_eclipse_info(),
        template_class=EclipseInfo,
        extraction_name="eclipse_info",
    )

    # Add helpful context and ground truth expectations (for transparency; not used for verification)
    evaluator.add_ground_truth({
        "expected_totality_duration": "58 minutes (approximately one hour)",
        "expected_constellation": "Leo",
        "note": "Verification is source-grounded against NASA official pages cited in the answer."
    })

    # Record custom info such as NASA URL count extracted
    nasa_urls = filter_nasa_urls(info.sources)
    evaluator.add_custom_info(
        info={
            "total_urls_extracted": len(info.sources),
            "nasa_urls_extracted": len(nasa_urls),
            "nasa_urls": nasa_urls
        },
        info_type="url_statistics"
    )

    # Build verification subtrees
    await verify_duration_of_totality(evaluator, root, info)
    await verify_constellation_location(evaluator, root, info)

    # Return structured evaluation summary
    return evaluator.get_summary()