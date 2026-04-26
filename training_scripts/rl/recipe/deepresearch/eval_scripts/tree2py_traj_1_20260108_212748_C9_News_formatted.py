import asyncio
import logging
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tech_news_2025_sectors"
TASK_DESCRIPTION = (
    "Identify four major technology news stories from 2025, each representing a different sector among: "
    "(1) AI/Tech M&A, (2) Cybersecurity Incidents, (3) Semiconductor Industry, and (4) Space Economy. "
    "For each story, provide the following information with verifiable sources: the technology sector it represents, "
    "the primary entity or entities involved (company names, organizations), a significant quantitative metric "
    "(financial value, number of incidents, market size, or achievement metric), the specific time period in 2025 "
    "when the event occurred or was reported, and a reference URL from a credible news source that confirms the "
    "information. Each story should represent a significant development in its respective sector and be supported by "
    "publicly available news reporting from 2025."
)

ALLOWED_SECTORS = [
    "AI/Tech M&A",
    "Cybersecurity Incidents",
    "Semiconductor Industry",
    "Space Economy",
]
TARGET_YEAR = 2025


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class StoryItem(BaseModel):
    sector: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    quantitative_metric: Optional[str] = None
    time_period_2025: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)
    significance: Optional[str] = None


class StoriesExtraction(BaseModel):
    stories: List[StoryItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_stories() -> str:
    return (
        "Extract up to four distinct technology news stories provided in the answer. For each story, return an object "
        "with the following fields strictly using these keys:\n"
        "- sector: the sector label explicitly mentioned in the answer for this story. Do NOT normalize; copy exactly "
        "  as stated in the answer (e.g., 'AI/Tech M&A', 'Cybersecurity Incidents', 'Semiconductor Industry', "
        "  'Space Economy', or close variants if that’s what the answer used).\n"
        "- entities: an array of the primary company/organization names (or market designations) that the answer "
        "  identifies as central to the story. Use exact names from the answer; no inference.\n"
        "- quantitative_metric: the significant quantitative metric text cited in the answer (e.g., '$62 billion deal', "
        "  '45% YoY', '23 incidents', 'market size $100B', 'launched 22 satellites'). Include the number and unit/"
        "  context as stated. If missing, set to null.\n"
        "- time_period_2025: the time period in 2025 mentioned in the answer for when the event occurred or was reported "
        "  (e.g., 'January 2025', 'Q2 2025', 'H1 2025', 'July 2025'). If missing or not in 2025, set to null.\n"
        "- source_urls: an array of all URLs given in the answer for this story. Extract only explicit URLs; if none, "
        "  return an empty array. Include full protocols.\n"
        "- significance: a short sentence from the answer indicating why this story is significant in its sector "
        "  (e.g., 'largest deal of the year', 'major breach affecting millions', 'record chip revenue', "
        "  'landmark launch'). If missing, set to null.\n\n"
        "Return a JSON object with a single field 'stories' that is an array of these story objects. "
        "Extract only what is explicitly present in the answer text; do not invent or infer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _stringify_entities(entities: List[str]) -> str:
    if not entities:
        return "None"
    return ", ".join([e.strip() for e in entities if e and e.strip()])


# --------------------------------------------------------------------------- #
# Per-story verification                                                      #
# --------------------------------------------------------------------------- #
async def verify_story(
    evaluator: Evaluator,
    parent_node,
    story: StoryItem,
    index: int,
) -> None:
    """
    Build verification subtree for a single story.
    """
    story_node = evaluator.add_parallel(
        id=f"story_{index+1}",
        desc=f"Story #{index+1} verification (sector, entities, metric, time period, sources, significance)",
        parent=parent_node,
        critical=False,
    )

    # 1) Sector (critical): presence and allowed category labeling (with synonyms allowed)
    sector_leaf = evaluator.add_leaf(
        id=f"story_{index+1}_sector",
        desc="Story explicitly labels which of the four allowed sectors it represents.",
        parent=story_node,
        critical=True,
    )
    sector_text = story.sector or "None"
    sector_claim = (
        f"In the answer, Story #{index+1} explicitly labels its sector as '{sector_text}'. "
        f"This label corresponds to one of the allowed categories: {', '.join(ALLOWED_SECTORS)} "
        f"(allow minor variations or synonyms such as 'AI M&A' for 'AI/Tech M&A', 'space industry' for 'Space Economy', "
        f"'semiconductors' for 'Semiconductor Industry')."
    )
    await evaluator.verify(
        claim=sector_claim,
        node=sector_leaf,
        additional_instruction=(
            "Use only the answer text to confirm the story presents a clear sector label and that it can be reasonably "
            "mapped to one of the allowed categories."
        ),
    )

    # 2) Primary entities (critical): identified in the answer text
    entities_leaf = evaluator.add_leaf(
        id=f"story_{index+1}_primary_entities",
        desc="Story identifies the primary entity or entities involved.",
        parent=story_node,
        critical=True,
    )
    entities_text = _stringify_entities(story.entities)
    entities_claim = (
        f"In the answer, Story #{index+1} identifies the following primary entities (companies/organizations): "
        f"{entities_text}. These are explicitly listed as central to the story."
    )
    await evaluator.verify(
        claim=entities_claim,
        node=entities_leaf,
        additional_instruction=(
            "Judge based solely on the answer text. The story should name the primary entities explicitly; "
            "do not validate against external sources in this leaf."
        ),
    )

    # 3) Quantitative metric (critical): includes number and unit/context
    metric_leaf = evaluator.add_leaf(
        id=f"story_{index+1}_quantitative_metric",
        desc="Story includes a significant quantitative metric with clear numeric value and unit/context.",
        parent=story_node,
        critical=True,
    )
    metric_text = story.quantitative_metric or "None"
    metric_claim = (
        f"Story #{index+1} includes the following significant quantitative metric: '{metric_text}'. "
        f"This contains at least one numeric value and a clear unit or contextual meaning "
        f"(e.g., $, %, count, market size, satellites, revenue, volume)."
    )
    await evaluator.verify(
        claim=metric_claim,
        node=metric_leaf,
        additional_instruction=(
            "Reject if the metric is missing, purely qualitative, or only a date. "
            "Accept rounding differences and common formatting (e.g., $62B vs $62 billion)."
        ),
    )

    # 4) Time period in 2025 (critical)
    time_leaf = evaluator.add_leaf(
        id=f"story_{index+1}_time_period_2025",
        desc="Story specifies the relevant time period in 2025 when the event occurred or was reported.",
        parent=story_node,
        critical=True,
    )
    time_text = story.time_period_2025 or "None"
    time_claim = (
        f"Story #{index+1} specifies a time period in 2025: '{time_text}'. "
        f"The period clearly falls within the year 2025 (e.g., a month, quarter, H1/H2, or 'in 2025')."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        additional_instruction=(
            "Check the answer text only for a 2025 time reference. Accept common formats such as 'Jan 2025', "
            "'Q2 2025', 'H1 2025', or explicit 'in 2025'."
        ),
    )

    # 5) Source URL presence (critical) - prerequisite for source-based checks
    urls_present = bool(story.source_urls)
    evaluator.add_custom_node(
        result=urls_present,
        id=f"story_{index+1}_source_urls_present",
        desc=f"Story #{index+1} provides at least one source URL.",
        parent=story_node,
        critical=True,
    )

    # 6) Source credibility (critical): at least one credible, publicly accessible outlet
    credible_leaf = evaluator.add_leaf(
        id=f"story_{index+1}_source_credible",
        desc="At least one provided URL is a credible publicly accessible news/industry source.",
        parent=story_node,
        critical=True,
    )
    credible_claim = (
        "At least one of the provided URLs is a credible, publicly accessible news/industry source "
        "(e.g., reputable news outlet, recognized trade publication, or official company newsroom/press release)."
    )
    await evaluator.verify(
        claim=credible_claim,
        node=credible_leaf,
        sources=story.source_urls,
        additional_instruction=(
            "Evaluate the credibility based on domain and page signals. Avoid anonymous blogs or low-credibility aggregators. "
            "Official corporate press rooms are acceptable."
        ),
    )

    # 7) Source support for entities (critical)
    entities_support_leaf = evaluator.add_leaf(
        id=f"story_{index+1}_source_supports_entities",
        desc="Source(s) substantiate the stated primary entities as central to the story.",
        parent=story_node,
        critical=True,
    )
    entities_support_claim = (
        f"The source(s) explicitly mention the primary entities as central to the reported event: {entities_text}."
    )
    await evaluator.verify(
        claim=entities_support_claim,
        node=entities_support_leaf,
        sources=story.source_urls,
        additional_instruction=(
            "Look for names and roles in the article(s). Minor name formatting differences are acceptable."
        ),
    )

    # 8) Source support for the quantitative metric (critical)
    metric_support_leaf = evaluator.add_leaf(
        id=f"story_{index+1}_source_supports_metric",
        desc="Source(s) substantiate the stated quantitative metric.",
        parent=story_node,
        critical=True,
    )
    metric_support_claim = (
        f"The source(s) report the stated quantitative metric: '{metric_text}'. "
        f"Allow rounding and formatting differences."
    )
    await evaluator.verify(
        claim=metric_support_claim,
        node=metric_support_leaf,
        sources=story.source_urls,
        additional_instruction=(
            "Confirm that the numeric value and its unit/context appear or are clearly supported in the article(s). "
            "Allow minor rounding (e.g., 62B vs 61.9B)."
        ),
    )

    # 9) Source mentions 2025 timing (critical)
    y2025_leaf = evaluator.add_leaf(
        id=f"story_{index+1}_source_mentions_2025",
        desc="Source(s) are from 2025 or clearly state the event/reporting occurred in 2025.",
        parent=story_node,
        critical=True,
    )
    y2025_claim = (
        "This page is a news item from 2025 or clearly states that the reported event occurred in 2025 "
        "(publication/updated date in 2025, or explicit reference to a 2025 period)."
    )
    await evaluator.verify(
        claim=y2025_claim,
        node=y2025_leaf,
        sources=story.source_urls,
        additional_instruction=(
            "Pass if the article shows a 2025 publication/update date or clearly describes the event timing in 2025. "
            "Fail if only a non-2025 date is present and no 2025 mention."
        ),
    )

    # 10) Significance indication (critical)
    significance_leaf = evaluator.add_leaf(
        id=f"story_{index+1}_significance",
        desc="Story includes a brief significance statement consistent with cited sources.",
        parent=story_node,
        critical=True,
    )
    significance_text = story.significance or "None"
    significance_claim = (
        f"Story #{index+1} includes a brief statement of significance: '{significance_text}', "
        f"and this significance is consistent with the cited source(s) (e.g., major deal/record impact/landmark event)."
    )
    await evaluator.verify(
        claim=significance_claim,
        node=significance_leaf,
        sources=story.source_urls,
        additional_instruction=(
            "Check that the significance described in the answer reasonably matches the emphasis or facts in the source(s)."
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
    Evaluate an answer against the rubric for four 2025 technology news stories across distinct sectors.
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

    # Extract structured stories from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_stories(),
        template_class=StoriesExtraction,
        extraction_name="stories_extraction",
    )

    # Sort/crop to the first four stories only (if more are present)
    stories: List[StoryItem] = list(extraction.stories or [])[:4]

    # Record useful custom info for debugging
    evaluator.add_custom_info(
        info={"extracted_story_count": len(extraction.stories or []),
              "used_story_count_for_eval": len(stories),
              "allowed_sectors": ALLOWED_SECTORS},
        info_type="debug_info",
        info_name="extraction_summary"
    )

    # ------------------ Global critical checks -------------------------- #
    global_checks = evaluator.add_parallel(
        id="global_checks",
        desc="Global constraints for four stories",
        parent=root,
        critical=True,
    )

    # Global story count must be exactly four and distinct
    global_count_leaf = evaluator.add_leaf(
        id="global_story_count",
        desc="Response provides exactly four distinct stories/items.",
        parent=global_checks,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer provides exactly four distinct stories/items (no more, no fewer).",
        node=global_count_leaf,
        additional_instruction=(
            "Judge by the answer text. Treat each top-level item/entry as one story. "
            "Do not count sub-points within a single story as separate items."
        ),
    )

    # Global sector coverage: four distinct sectors covering the allowed set
    sectors_in_answer = [s.sector or "None" for s in stories]
    coverage_leaf = evaluator.add_leaf(
        id="global_sector_coverage",
        desc=(
            "Across the four stories, sectors are all distinct and collectively cover: "
            "AI/Tech M&A, Cybersecurity Incidents, Semiconductor Industry, and Space Economy."
        ),
        parent=global_checks,
        critical=True,
    )
    coverage_claim = (
        f"The four sector labels in the answer are: {sectors_in_answer}. "
        f"Collectively they represent four distinct sectors, covering each of the four target categories: "
        f"{', '.join(ALLOWED_SECTORS)}. Allow reasonable synonyms or formatting variations."
    )
    await evaluator.verify(
        claim=coverage_claim,
        node=coverage_leaf,
        additional_instruction=(
            "Map reasonable variants to the target categories (e.g., 'AI M&A' ~ 'AI/Tech M&A', "
            "'semiconductors' ~ 'Semiconductor Industry', 'space industry' ~ 'Space Economy')."
        ),
    )

    # ------------------ Per-story verification (non-critical as a group) ------------------ #
    stories_root = evaluator.add_parallel(
        id="stories",
        desc="Per-story verification",
        parent=root,
        critical=False,
    )

    # Build four per-story subtrees (use placeholders if fewer than four)
    for i in range(4):
        story = stories[i] if i < len(stories) else StoryItem()
        await verify_story(evaluator, stories_root, story, i)

    # Add ground-truth reference info (allowed sectors and target year)
    evaluator.add_ground_truth(
        gt_info={
            "allowed_sectors": ALLOWED_SECTORS,
            "target_year": TARGET_YEAR,
            "require_exactly_four_stories": True,
            "distinct_sectors_required": True,
        },
        gt_type="rubric_expectations",
    )

    return evaluator.get_summary()