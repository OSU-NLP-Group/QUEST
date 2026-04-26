import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "slc_ski_resort_largest_acres_qualifiers"
TASK_DESCRIPTION = (
    "Identify the ski resort near Salt Lake City International Airport that meets ALL of the following criteria: "
    "(1) Located within 35 miles of Salt Lake City International Airport, "
    "(2) Allows snowboarding (not a skiers-only resort), "
    "(3) Has a vertical drop of at least 2,500 feet, "
    "(4) Has a summit elevation of at least 10,500 feet, "
    "(5) Among all resorts meeting criteria 1-4, has the LARGEST skiable terrain measured in acres. "
    "Provide the following information about this resort: resort name, exact skiable terrain size in acres, exact vertical drop in feet, "
    "exact summit elevation in feet, base elevation in feet, distance from Salt Lake City International Airport in miles, "
    "and at least one reference URL confirming these specifications."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResortExtraction(BaseModel):
    resort_name: Optional[str] = None
    terrain_acres: Optional[str] = None
    vertical_drop_ft: Optional[str] = None
    summit_elevation_ft: Optional[str] = None
    base_elevation_ft: Optional[str] = None
    distance_from_slc_miles: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resort() -> str:
    return """
    From the answer, extract the single ski resort the answer identifies as fulfilling ALL required criteria and being the final choice.
    Return the following fields (use strings for all numeric values as they appear, do not convert units):
    - resort_name: the resort's name as written in the answer.
    - terrain_acres: the exact skiable terrain size in acres as written (e.g., "2500", "2,500", "2,500+").
    - vertical_drop_ft: the exact vertical drop in feet as written (e.g., "3240 ft", "3,240 feet").
    - summit_elevation_ft: the exact summit elevation in feet as written (e.g., "11,000 ft").
    - base_elevation_ft: the base elevation in feet as written (if provided).
    - distance_from_slc_miles: the distance in miles from Salt Lake City International Airport as written (if provided). If the answer only gives minutes, return null.
    - reference_urls: a list of all URLs the answer provides to support the choice and specifications for this resort. Include only fully formed HTTP/HTTPS URLs that appear in the answer (plain or markdown links).
    If any field is missing in the answer, set it to null (or [] for the list of URLs).
    Only extract the single resort that is claimed to meet all criteria and be the final choice; do not extract multiple resorts.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def filter_valid_urls(urls: List[str]) -> List[str]:
    out = []
    for u in urls:
        if isinstance(u, str) and (u.strip().lower().startswith("http://") or u.strip().lower().startswith("https://")):
            out.append(u.strip())
    return out


# --------------------------------------------------------------------------- #
# Verification subtree builder                                                #
# --------------------------------------------------------------------------- #
async def build_and_verify_resort_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: ResortExtraction,
) -> None:
    """
    Construct the verification tree according to the rubric and run verifications.
    """

    group = evaluator.add_parallel(
        id="Ski_Resort_Identification",
        desc="Correct identification of the ski resort near SLC with the largest skiable terrain among resorts that meet the constraints",
        parent=parent_node,
        critical=False  # Parent non-critical to allow partial credit per framework constraint
    )

    # Pre-check for sources (critical in rubric)
    valid_urls = filter_valid_urls(extracted.reference_urls or [])
    has_sources = len(valid_urls) > 0
    ref_url_node = evaluator.add_custom_node(
        result=has_sources,
        id="Reference_URL",
        desc="The answer includes at least one valid reference URL supporting the identification and specifications of the resort",
        parent=group,
        critical=True
    )

    resort_name = extracted.resort_name or ""

    # Resort Name (critical) - verify the name is supported by provided sources
    name_node = evaluator.add_leaf(
        id="Resort_Name",
        desc="The answer provides the correct name of the identified ski resort",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ski resort described by the provided source pages is named '{resort_name}'. "
              f"Minor naming variants (e.g., adding 'Ski Resort', 'Mountain', or punctuation) should still count as a match.",
        node=name_node,
        sources=valid_urls if has_sources else None,
        additional_instruction="Focus on whether the pages clearly identify the resort with the provided name or an equivalent variant. Treat case, hyphenation, and 'Ski Resort/Mountain' suffixes as acceptable variants.",
        extra_prerequisites=[ref_url_node]
    )

    # Distance within 35 miles (critical)
    distance_within_node = evaluator.add_leaf(
        id="Distance_Within_35_Miles",
        desc="The identified resort is verified to be within 35 miles of SLC airport",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{resort_name} is located within 35 miles of Salt Lake City International Airport (SLC).",
        node=distance_within_node,
        sources=valid_urls if has_sources else None,
        additional_instruction=(
            "Look for an explicit mileage figure from SLC Airport or the city/airport to the resort. "
            "If the pages only provide approximate drive time and not miles, do not infer miles unless a distance in miles is also provided. "
            "If no page supports the within-35-miles claim, return Not Supported."
        ),
        extra_prerequisites=[ref_url_node]
    )

    # Allows Snowboarding (critical)
    snowboard_node = evaluator.add_leaf(
        id="Allows_Snowboarding",
        desc="The identified resort is confirmed to allow snowboarding (not skiers-only)",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{resort_name} allows snowboarding (it is not a skiers-only resort).",
        node=snowboard_node,
        sources=valid_urls if has_sources else None,
        additional_instruction="Find explicit statements like 'ski and snowboard' or policies that confirm snowboarding is allowed. If pages indicate 'skiers only', it should fail.",
        extra_prerequisites=[ref_url_node]
    )

    # Vertical Drop requirement (critical)
    vertical_req_node = evaluator.add_leaf(
        id="Vertical_Drop_Requirement",
        desc="The identified resort has a vertical drop of at least 2,500 feet",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The vertical drop of {resort_name} is at least 2,500 feet.",
        node=vertical_req_node,
        sources=valid_urls if has_sources else None,
        additional_instruction="Use official stats or reputable ski data pages. If multiple values appear, accept if any source credibly supports ≥ 2,500 ft.",
        extra_prerequisites=[ref_url_node]
    )

    # Summit Elevation requirement (critical)
    summit_req_node = evaluator.add_leaf(
        id="Summit_Elevation_Requirement",
        desc="The identified resort has a summit elevation of at least 10,500 feet",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The summit elevation of {resort_name} is at least 10,500 feet.",
        node=summit_req_node,
        sources=valid_urls if has_sources else None,
        additional_instruction="Use official stats or reputable ski data pages. If multiple values appear, accept if any source credibly supports ≥ 10,500 ft.",
        extra_prerequisites=[ref_url_node]
    )

    # Largest qualifying terrain among those meeting 1-4 (critical)
    largest_node = evaluator.add_leaf(
        id="Largest_Qualifying_Terrain",
        desc="Among all resorts meeting the above criteria, the identified resort has the largest skiable terrain",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"Among ski resorts within 35 miles of Salt Lake City International Airport that allow snowboarding, "
            f"have a vertical drop of at least 2,500 feet, and a summit elevation of at least 10,500 feet, "
            f"{resort_name} has the largest skiable terrain (in acres)."
        ),
        node=largest_node,
        sources=valid_urls if has_sources else None,
        additional_instruction=(
            "Use the provided URLs collectively. This claim must be explicitly or implicitly supported (e.g., "
            "a page states this resort has the largest terrain among nearby qualifying resorts or shows comparative stats that imply it). "
            "If the provided pages do not support this superlative relative to other qualifying resorts, return Not Supported."
        ),
        extra_prerequisites=[ref_url_node]
    )

    # Exact Terrain Size (non-critical)
    terrain_exact_node = evaluator.add_leaf(
        id="Exact_Terrain_Size",
        desc="The answer provides the exact skiable acreage of the identified resort",
        parent=group,
        critical=False
    )
    await evaluator.verify(
        claim=f"The skiable terrain at {resort_name} is exactly '{extracted.terrain_acres or ''}' acres (allow minor rounding/formatting).",
        node=terrain_exact_node,
        sources=valid_urls if has_sources else None,
        additional_instruction="Verify the acreage number appears on one of the sources. Allow commas, '+' suffixes, or small rounding (e.g., 2,500 vs 2,500+).",
        extra_prerequisites=[ref_url_node]
    )

    # Exact Vertical Drop (non-critical)
    vertical_exact_node = evaluator.add_leaf(
        id="Exact_Vertical_Drop",
        desc="The answer provides the exact vertical drop measurement of the identified resort",
        parent=group,
        critical=False
    )
    await evaluator.verify(
        claim=f"The vertical drop at {resort_name} is exactly '{extracted.vertical_drop_ft or ''}' (in feet; allow minor formatting).",
        node=vertical_exact_node,
        sources=valid_urls if has_sources else None,
        additional_instruction="Match the stated number on the provided page(s). Allow commas and the words 'ft' or 'feet'.",
        extra_prerequisites=[ref_url_node]
    )

    # Exact Summit Elevation (non-critical)
    summit_exact_node = evaluator.add_leaf(
        id="Exact_Summit_Elevation",
        desc="The answer provides the exact summit elevation of the identified resort",
        parent=group,
        critical=False
    )
    await evaluator.verify(
        claim=f"The summit elevation at {resort_name} is exactly '{extracted.summit_elevation_ft or ''}' (in feet; allow minor formatting).",
        node=summit_exact_node,
        sources=valid_urls if has_sources else None,
        additional_instruction="Match the stated summit elevation. Allow commas and the words 'ft' or 'feet'.",
        extra_prerequisites=[ref_url_node]
    )

    # Base Elevation (non-critical)
    base_exact_node = evaluator.add_leaf(
        id="Base_Elevation",
        desc="The answer provides the base elevation of the identified resort",
        parent=group,
        critical=False
    )
    await evaluator.verify(
        claim=f"The base elevation at {resort_name} is exactly '{extracted.base_elevation_ft or ''}' (in feet; allow minor formatting).",
        node=base_exact_node,
        sources=valid_urls if has_sources else None,
        additional_instruction="Match the stated base elevation on a provided source. Allow commas and 'ft/feet'. If not present on sources, mark Not Supported.",
        extra_prerequisites=[ref_url_node]
    )

    # Exact Distance (non-critical)
    distance_exact_node = evaluator.add_leaf(
        id="Exact_Distance",
        desc="The answer provides the specific distance from SLC airport to the identified resort",
        parent=group,
        critical=False
    )
    await evaluator.verify(
        claim=f"The distance from Salt Lake City International Airport (SLC) to {resort_name} is exactly '{extracted.distance_from_slc_miles or ''}' miles (allow minor rounding).",
        node=distance_exact_node,
        sources=valid_urls if has_sources else None,
        additional_instruction="Only accept if a provided page explicitly gives miles from SLC (airport or city). If only minutes are given with no mileage, do not count as supported.",
        extra_prerequisites=[ref_url_node]
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
    Evaluate an answer for the SLC ski resort identification with constraints and largest qualifying terrain.
    """
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

    # Extract structured resort info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_resort(),
        template_class=ResortExtraction,
        extraction_name="resort_extraction",
    )

    # Store some helpful custom info
    evaluator.add_custom_info(
        info={
            "resort_name": extracted.resort_name,
            "terrain_acres": extracted.terrain_acres,
            "vertical_drop_ft": extracted.vertical_drop_ft,
            "summit_elevation_ft": extracted.summit_elevation_ft,
            "base_elevation_ft": extracted.base_elevation_ft,
            "distance_from_slc_miles": extracted.distance_from_slc_miles,
            "total_reference_urls_found": len(extracted.reference_urls or []),
            "valid_reference_urls": filter_valid_urls(extracted.reference_urls or []),
        },
        info_type="parsed_answer_overview",
    )

    # Build and verify tree
    await build_and_verify_resort_tree(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()