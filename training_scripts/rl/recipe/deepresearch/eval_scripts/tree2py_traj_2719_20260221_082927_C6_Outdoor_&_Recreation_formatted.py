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
TASK_ID = "bc_co_ski_resort"
TASK_DESCRIPTION = """
Identify a ski resort in North America (specifically in British Columbia, Canada or Colorado, USA) that meets all of the following criteria: (1) has at least 5,000 acres of skiable terrain, (2) offers a vertical drop of at least 4,000 feet, (3) has at least 30 operational lifts, (4) has at least 150 marked trails, (5) operates at least 2 terrain parks, (6) has a base elevation above 650 meters (2,130 feet), (7) has a top elevation exceeding 2,100 meters (6,890 feet), (8) receives an average annual snowfall of at least 400 inches, (9) offers an adaptive skiing program for individuals with disabilities, (10) has documented sustainability initiatives or environmental certifications, and (11) offers summer lift-accessed activities. Provide the resort name, and for each criterion, include the specific factual information and a reference URL that supports your answer.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResortExtraction(BaseModel):
    """Structured extraction of resort info and supporting URLs from the answer."""
    resort_name: Optional[str] = None
    resort_identity_urls: List[str] = Field(default_factory=list)

    location: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    acres: Optional[str] = None
    acres_urls: List[str] = Field(default_factory=list)

    vertical_drop: Optional[str] = None
    vertical_drop_urls: List[str] = Field(default_factory=list)

    lifts: Optional[str] = None
    lifts_urls: List[str] = Field(default_factory=list)

    trails: Optional[str] = None
    trails_urls: List[str] = Field(default_factory=list)

    terrain_parks: Optional[str] = None
    terrain_parks_urls: List[str] = Field(default_factory=list)

    base_elevation: Optional[str] = None
    base_elev_urls: List[str] = Field(default_factory=list)

    summit_elevation: Optional[str] = None
    summit_elev_urls: List[str] = Field(default_factory=list)

    average_snowfall: Optional[str] = None
    snowfall_urls: List[str] = Field(default_factory=list)

    adaptive_program_desc: Optional[str] = None
    adaptive_urls: List[str] = Field(default_factory=list)

    sustainability_desc: Optional[str] = None
    sustainability_urls: List[str] = Field(default_factory=list)

    summer_activities_desc: Optional[str] = None
    summer_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resort() -> str:
    return """
    Extract the ski resort information presented in the answer. Return a JSON object with the following fields:

    1) resort_name: The specific resort name mentioned.
    2) resort_identity_urls: A list of URLs cited in the answer that confirm the resort's identity (e.g., official site, Wikipedia, etc.).

    3) location: The resort's location as stated (e.g., 'British Columbia, Canada' or 'Colorado, USA').
    4) location_urls: A list of URLs cited that specifically support the location.

    5) acres: The stated skiable terrain figure (e.g., '8,171 acres').
    6) acres_urls: A list of URLs cited that support the acreage claim.

    7) vertical_drop: The stated vertical drop measurement (e.g., '5,000 ft').
    8) vertical_drop_urls: A list of URLs cited that support the vertical drop claim.

    9) lifts: The stated number of operational lifts (e.g., '36 lifts').
    10) lifts_urls: A list of URLs cited that support the lifts count claim.

    11) trails: The stated number of marked trails (e.g., '200 trails').
    12) trails_urls: A list of URLs cited that support the trails count claim.

    13) terrain_parks: The stated number of terrain parks (e.g., '3 terrain parks').
    14) terrain_parks_urls: A list of URLs cited that support the terrain parks claim.

    15) base_elevation: The stated base elevation (e.g., '675 m' or '2,215 ft').
    16) base_elev_urls: A list of URLs cited that support the base elevation claim.

    17) summit_elevation: The stated top/summit elevation (e.g., '2,300 m' or '7,546 ft').
    18) summit_elev_urls: A list of URLs cited that support the summit elevation claim.

    19) average_snowfall: The stated average annual snowfall (e.g., '420 inches annually').
    20) snowfall_urls: A list of URLs cited that support the snowfall claim.

    21) adaptive_program_desc: A short phrase confirming an adaptive skiing program for individuals with disabilities (e.g., 'Adaptive Ski Program').
    22) adaptive_urls: A list of URLs cited that support the adaptive program claim (e.g., resort's adaptive lessons page, partner org page).

    23) sustainability_desc: A short phrase identifying sustainability initiatives or environmental certifications (e.g., 'STOKE Certified', 'Sustainability program', etc.).
    24) sustainability_urls: A list of URLs cited that support sustainability claims.

    25) summer_activities_desc: A short phrase confirming summer lift-accessed activities (e.g., 'lift-access mountain biking', 'scenic gondola rides').
    26) summer_urls: A list of URLs cited that support the summer operations claim.

    IMPORTANT:
    - Extract only information explicitly present in the answer.
    - For each URL list field, return all URLs explicitly cited for that claim. If no URLs are cited, return an empty array.
    - URLs may be plain or in markdown; extract the actual URLs.
    - Do not invent any values or URLs. If a value is not mentioned, set it to null; for URLs, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any((u or "").strip() for u in urls)


async def verify_by_urls_or_fail(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    add_ins: str,
    critical: bool = True
) -> None:
    """
    Create a leaf node and verify by URLs. If URLs are missing, mark leaf as failed (no verification).
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )
    if not has_urls(urls):
        # Enforce source-grounding: missing URLs -> fail
        leaf.score = 0.0
        leaf.status = "failed"
        return
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_resort_identification(evaluator: Evaluator, parent_node, ext: ResortExtraction) -> None:
    node = evaluator.add_parallel(
        id="resort_identification",
        desc="Resort name is clearly identified",
        parent=parent_node,
        critical=True
    )

    # Leaf: resort name provided (existence)
    evaluator.add_custom_node(
        result=(ext.resort_name is not None and ext.resort_name.strip() != ""),
        id="resort_name_provided",
        desc="Specific resort name is stated",
        parent=node,
        critical=True
    )

    # Leaf: URL confirming resort identity
    await verify_by_urls_or_fail(
        evaluator=evaluator,
        parent_node=node,
        node_id="resort_name_url",
        desc="URL reference provided confirming resort identity",
        claim=f"The page identifies a ski resort named '{ext.resort_name or ''}'.",
        urls=ext.resort_identity_urls,
        add_ins="Verify that the webpage clearly names the resort with that exact or very similar name (allow minor variations and punctuation)."
    )


async def build_terrain_infrastructure(evaluator: Evaluator, parent_node, ext: ResortExtraction) -> None:
    node = evaluator.add_parallel(
        id="terrain_infrastructure",
        desc="Resort meets terrain size and infrastructure requirements",
        parent=parent_node,
        critical=True
    )

    # Acres
    acres_node = evaluator.add_parallel(
        id="terrain_size_requirement",
        desc="Resort has at least 5,000 acres of skiable terrain",
        parent=node,
        critical=True
    )
    await verify_by_urls_or_fail(
        evaluator, acres_node, "terrain_size_value",
        "Specific acreage figure stated and equals or exceeds 5,000 acres",
        "The resort has at least 5,000 acres of skiable terrain.",
        ext.acres_urls,
        add_ins=f"The answer states acreage as '{ext.acres}'. Confirm the page supports a skiable terrain area ≥ 5,000 acres. Allow unit conversions if metric is used."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.acres_urls),
        id="terrain_size_url",
        desc="URL reference provided for terrain size claim",
        parent=acres_node,
        critical=True
    )

    # Vertical drop
    vd_node = evaluator.add_parallel(
        id="vertical_drop_requirement",
        desc="Resort offers vertical drop of at least 4,000 feet",
        parent=node,
        critical=True
    )
    await verify_by_urls_or_fail(
        evaluator, vd_node, "vertical_drop_value",
        "Specific vertical drop measurement stated and equals or exceeds 4,000 feet",
        "The resort has a vertical drop of at least 4,000 feet.",
        ext.vertical_drop_urls,
        add_ins=f"The answer states vertical drop as '{ext.vertical_drop}'. Confirm the page supports ≥ 4,000 ft (or ≈ 1,219 m). Allow metric-imperial conversions."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.vertical_drop_urls),
        id="vertical_drop_url",
        desc="URL reference provided for vertical drop claim",
        parent=vd_node,
        critical=True
    )

    # Lifts
    lifts_node = evaluator.add_parallel(
        id="lift_capacity_requirement",
        desc="Resort has at least 30 operational lifts",
        parent=node,
        critical=True
    )
    await verify_by_urls_or_fail(
        evaluator, lifts_node, "lift_count_value",
        "Specific number of lifts stated and equals or exceeds 30",
        "The resort operates at least 30 lifts.",
        ext.lifts_urls,
        add_ins=f"The answer states lifts as '{ext.lifts}'. Confirm the page indicates ≥ 30 operational lifts (include chairs, gondolas, surface lifts if counted)."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.lifts_urls),
        id="lift_count_url",
        desc="URL reference provided for lift count claim",
        parent=lifts_node,
        critical=True
    )

    # Trails
    trails_node = evaluator.add_parallel(
        id="trail_variety_requirement",
        desc="Resort has at least 150 marked trails",
        parent=node,
        critical=True
    )
    await verify_by_urls_or_fail(
        evaluator, trails_node, "trail_count_value",
        "Specific number of trails stated and equals or exceeds 150",
        "The resort has at least 150 marked trails.",
        ext.trails_urls,
        add_ins=f"The answer states trails as '{ext.trails}'. Confirm the page supports ≥ 150 marked trails/runs (allow synonyms like 'runs')."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.trails_urls),
        id="trail_count_url",
        desc="URL reference provided for trail count claim",
        parent=trails_node,
        critical=True
    )

    # Terrain parks
    parks_node = evaluator.add_parallel(
        id="terrain_park_requirement",
        desc="Resort operates at least 2 terrain parks",
        parent=node,
        critical=True
    )
    await verify_by_urls_or_fail(
        evaluator, parks_node, "terrain_park_value",
        "Specific number of terrain parks stated and equals or exceeds 2",
        "The resort operates at least 2 terrain parks.",
        ext.terrain_parks_urls,
        add_ins=f"The answer states terrain parks as '{ext.terrain_parks}'. Confirm the page indicates ≥ 2 terrain parks (accept synonyms 'terrain park', 'snow park', 'freestyle park')."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.terrain_parks_urls),
        id="terrain_park_url",
        desc="URL reference provided for terrain park claim",
        parent=parks_node,
        critical=True
    )


async def build_elevation_specs(evaluator: Evaluator, parent_node, ext: ResortExtraction) -> None:
    node = evaluator.add_parallel(
        id="elevation_specifications",
        desc="Resort meets elevation requirements for base and summit",
        parent=parent_node,
        critical=True
    )

    # Base elevation
    base_node = evaluator.add_parallel(
        id="base_elevation_requirement",
        desc="Resort base elevation exceeds 650 meters (2,130 feet)",
        parent=node,
        critical=True
    )
    await verify_by_urls_or_fail(
        evaluator, base_node, "base_elevation_value",
        "Specific base elevation stated and exceeds 650 meters or 2,130 feet",
        "The resort base elevation is above 650 meters (2,130 ft).",
        ext.base_elev_urls,
        add_ins=f"The answer states base elevation as '{ext.base_elevation}'. Confirm the page supports a base elevation > 650 m (≈ 2,130 ft). Allow unit conversions."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.base_elev_urls),
        id="base_elevation_url",
        desc="URL reference provided for base elevation claim",
        parent=base_node,
        critical=True
    )

    # Summit elevation
    summit_node = evaluator.add_parallel(
        id="summit_elevation_requirement",
        desc="Resort top elevation exceeds 2,100 meters (6,890 feet)",
        parent=node,
        critical=True
    )
    await verify_by_urls_or_fail(
        evaluator, summit_node, "summit_elevation_value",
        "Specific summit elevation stated and exceeds 2,100 meters or 6,890 feet",
        "The resort top/summit elevation exceeds 2,100 meters (6,890 ft).",
        ext.summit_elev_urls,
        add_ins=f"The answer states summit elevation as '{ext.summit_elevation}'. Confirm the page supports a summit/top elevation > 2,100 m (≈ 6,890 ft). Allow conversions."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.summit_elev_urls),
        id="summit_elevation_url",
        desc="URL reference provided for summit elevation claim",
        parent=summit_node,
        critical=True
    )

    # Snowfall
    snow_node = evaluator.add_parallel(
        id="snowfall_requirement",
        desc="Resort receives average annual snowfall of at least 400 inches",
        parent=node,
        critical=True
    )
    await verify_by_urls_or_fail(
        evaluator, snow_node, "snowfall_value",
        "Specific annual snowfall figure stated and equals or exceeds 400 inches",
        "The resort receives an average annual snowfall of at least 400 inches.",
        ext.snowfall_urls,
        add_ins=f"The answer states average snowfall as '{ext.average_snowfall}'. Confirm the page supports an average ≥ 400 inches annually. Distinguish annual average from single-season anomalies."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.snowfall_urls),
        id="snowfall_url",
        desc="URL reference provided for snowfall claim",
        parent=snow_node,
        critical=True
    )


async def build_accessibility_programs(evaluator: Evaluator, parent_node, ext: ResortExtraction) -> None:
    node = evaluator.add_parallel(
        id="accessibility_programs",
        desc="Resort offers adaptive skiing program for individuals with disabilities",
        parent=parent_node,
        critical=True
    )

    await verify_by_urls_or_fail(
        evaluator, node, "adaptive_program_value",
        "Specific adaptive skiing program described or confirmed to exist",
        "The resort offers an adaptive skiing program for individuals with disabilities.",
        ext.adaptive_urls,
        add_ins=f"The answer references '{ext.adaptive_program_desc}'. Verify existence of adaptive lessons/programs (terms like 'adaptive', 'sit-ski', 'para skiing', 'lessons for disabled')."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.adaptive_urls),
        id="adaptive_program_url",
        desc="URL reference provided for adaptive program claim",
        parent=node,
        critical=True
    )


async def build_environmental_commitment(evaluator: Evaluator, parent_node, ext: ResortExtraction) -> None:
    node = evaluator.add_parallel(
        id="environmental_commitment",
        desc="Resort has documented sustainability initiatives or environmental certifications",
        parent=parent_node,
        critical=True
    )

    await verify_by_urls_or_fail(
        evaluator, node, "sustainability_value",
        "Specific sustainability programs, initiatives, or certifications identified",
        "The resort has documented sustainability initiatives or environmental certifications.",
        ext.sustainability_urls,
        add_ins=f"The answer references '{ext.sustainability_desc}'. Verify programs/certifications (e.g., STOKE, ISO 14001, climate action plans, renewable energy, waste/recycling initiatives)."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.sustainability_urls),
        id="sustainability_url",
        desc="URL reference provided for sustainability claim",
        parent=node,
        critical=True
    )


async def build_summer_operations(evaluator: Evaluator, parent_node, ext: ResortExtraction) -> None:
    node = evaluator.add_parallel(
        id="summer_operations",
        desc="Resort offers summer lift-accessed activities",
        parent=parent_node,
        critical=True
    )

    await verify_by_urls_or_fail(
        evaluator, node, "summer_activities_value",
        "Specific summer activities and lift operations confirmed",
        "The resort offers summer lift-accessed activities (e.g., scenic rides, mountain biking).",
        ext.summer_urls,
        add_ins=f"The answer references '{ext.summer_activities_desc}'. Confirm that lifts operate in summer for activities such as biking, hiking, sightseeing (look for 'summer ops', 'bike park', 'scenic gondola')."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.summer_urls),
        id="summer_activities_url",
        desc="URL reference provided for summer operations claim",
        parent=node,
        critical=True
    )


async def build_geographic_location(evaluator: Evaluator, parent_node, ext: ResortExtraction) -> None:
    node = evaluator.add_parallel(
        id="geographic_location",
        desc="Resort is located in British Columbia, Canada or Colorado, USA",
        parent=parent_node,
        critical=True
    )

    # Verify location claim via URLs
    await verify_by_urls_or_fail(
        evaluator, node, "location_value",
        "Resort location confirmed to be in British Columbia or Colorado",
        "The resort is located in British Columbia, Canada or Colorado, USA.",
        ext.location_urls,
        add_ins=f"The answer states location as '{ext.location}'. Verify province/state is either British Columbia (BC) or Colorado (CO). Allow synonyms/abbreviations ('BC', 'CO')."
    )
    evaluator.add_custom_node(
        result=has_urls(ext.location_urls),
        id="location_url",
        desc="URL reference provided for location claim",
        parent=node,
        critical=True
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
    Evaluate an answer for the BC/CO ski resort comprehensive criteria task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify a North American ski resort meeting comprehensive criteria across terrain, infrastructure, accessibility, sustainability, and operational dimensions",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Create a critical task node under framework root to mirror rubric's critical root
    task_main = evaluator.add_parallel(
        id="task_main",
        desc="Identify a North American ski resort meeting comprehensive criteria across terrain, infrastructure, accessibility, sustainability, and operational dimensions",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    ext = await evaluator.extract(
        prompt=prompt_extract_resort(),
        template_class=ResortExtraction,
        extraction_name="resort_extraction",
    )

    # Build verification subtrees according to rubric
    await build_resort_identification(evaluator, task_main, ext)
    await build_terrain_infrastructure(evaluator, task_main, ext)
    await build_elevation_specs(evaluator, task_main, ext)
    await build_accessibility_programs(evaluator, task_main, ext)
    await build_environmental_commitment(evaluator, task_main, ext)
    await build_summer_operations(evaluator, task_main, ext)
    await build_geographic_location(evaluator, task_main, ext)

    # Return structured summary
    return evaluator.get_summary()