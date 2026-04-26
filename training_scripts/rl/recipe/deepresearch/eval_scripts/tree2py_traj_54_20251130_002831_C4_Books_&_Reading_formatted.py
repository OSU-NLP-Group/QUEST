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
TASK_ID = "cookbook_2016_grilling_130"
TASK_DESCRIPTION = (
    "Identify the cookbook and its author that satisfy ALL of the following criteria: "
    "(1) The cookbook was published in 2016, "
    "(2) It contains exactly 130 recipes, "
    "(3) It focuses specifically on outdoor cooking and grilling adventures, "
    "(4) The book has 352 pages, "
    "(5) The author is a television personality, "
    "(6) The author has restaurant locations in both California and Nevada, "
    "(7) The cookbook was published by William Morrow Cookbooks. "
    "Provide the exact title of the cookbook and the name of the author."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CookbookExtraction(BaseModel):
    """
    Extract the cookbook identification info as stated in the answer, plus all URLs
    explicitly cited by the answer for evidence verification.
    """
    title: Optional[str] = None
    author: Optional[str] = None

    # Optional fields if the answer states them explicitly (not required for verification,
    # but extracted for reference).
    stated_publication_year: Optional[str] = None
    stated_recipe_count: Optional[str] = None
    stated_page_count: Optional[str] = None
    stated_publisher: Optional[str] = None
    stated_theme_summary: Optional[str] = None

    # Evidence cited in the answer (URLs only, do not infer)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cookbook_info() -> str:
    return """
    Extract the cookbook identification information exactly as presented in the answer, and all URLs explicitly cited.

    Required fields:
    - title: The exact title of the cookbook stated in the answer.
    - author: The name of the author stated in the answer.

    Optional fields (if the answer states them explicitly):
    - stated_publication_year: The publication year as stated in the answer (e.g., "2016"). If not stated, return null.
    - stated_recipe_count: The number of recipes as stated (e.g., "130" or "130 recipes"). If not stated, return null.
    - stated_page_count: The page count as stated (e.g., "352 pages"). If not stated, return null.
    - stated_publisher: The publisher name as stated (e.g., "William Morrow Cookbooks"). If not stated, return null.
    - stated_theme_summary: A short phrase or sentence describing the book's focus/theme as stated (e.g., "outdoor cooking and grilling adventures"). If not stated, return null.

    URLs extraction (very important):
    - source_urls: Collect ALL URLs explicitly present in the answer that relate to this cookbook or its author.
      Examples include product pages (publisher site, Amazon, Google Books), author official site, Wikipedia, restaurant pages, press releases, etc.
      Return only valid URLs. Do not invent or infer any URL not present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _prep_sources(extracted: CookbookExtraction) -> List[str]:
    # Deduplicate, basic cleanup
    seen = set()
    result = []
    for url in extracted.source_urls:
        if not _non_empty(url):
            continue
        u = url.strip()
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


async def build_and_verify_cookbook_nodes(
    evaluator: Evaluator,
    parent_node,
    info: CookbookExtraction,
) -> None:
    """
    Build the verification subtree under the critical 'Cookbook_Identification' node and
    verify each required constraint using the URLs cited by the answer when available.
    """
    # Create the main critical node for all constraints
    main_node = evaluator.add_parallel(
        id="Cookbook_Identification",
        desc="Correctly identify the cookbook and its author that satisfy all specified constraints",
        parent=parent_node,
        critical=True
    )

    sources = _prep_sources(info)

    # Create all leaf nodes first
    node_title = evaluator.add_leaf(
        id="Title_Identification",
        desc="Provide the exact title of the cookbook",
        parent=main_node,
        critical=True
    )
    node_author = evaluator.add_leaf(
        id="Author_Name",
        desc="Provide the name of the author",
        parent=main_node,
        critical=True
    )
    node_pub_year = evaluator.add_leaf(
        id="Publication_Year",
        desc="The cookbook was published in 2016",
        parent=main_node,
        critical=True
    )
    node_recipe_count = evaluator.add_leaf(
        id="Recipe_Count",
        desc="The cookbook contains exactly 130 recipes",
        parent=main_node,
        critical=True
    )
    node_theme = evaluator.add_leaf(
        id="Content_Theme",
        desc="The cookbook focuses on outdoor cooking and grilling adventures",
        parent=main_node,
        critical=True
    )
    node_page_count = evaluator.add_leaf(
        id="Page_Count",
        desc="The cookbook has 352 pages",
        parent=main_node,
        critical=True
    )
    node_author_bg = evaluator.add_leaf(
        id="Author_Background",
        desc="The author is a television personality",
        parent=main_node,
        critical=True
    )
    node_restaurant_states = evaluator.add_leaf(
        id="Restaurant_Locations",
        desc="The author has restaurant locations in both California and Nevada",
        parent=main_node,
        critical=True
    )
    node_publisher = evaluator.add_leaf(
        id="Publisher",
        desc="The cookbook was published by William Morrow Cookbooks",
        parent=main_node,
        critical=True
    )

    # Build claims. For nodes requiring title/author, fail early if missing.
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Title verification
    if _non_empty(info.title):
        claims_and_sources.append((
            f"The exact title of the cookbook is '{info.title.strip()}'.",
            sources if sources else None,
            node_title,
            "Confirm the main book title as shown on the cited pages. Allow minor punctuation/case variations but the semantic title must match the book page title."
        ))
    else:
        node_title.score = 0.0
        node_title.status = "failed"

    # Author verification
    if _non_empty(info.author):
        title_context = info.title.strip() if _non_empty(info.title) else "this cookbook"
        claims_and_sources.append((
            f"The author of {title_context} is '{info.author.strip()}'.",
            sources if sources else None,
            node_author,
            "Verify the named author on the cited pages. If multiple contributors exist, the primary author must match the provided name."
        ))
    else:
        node_author.score = 0.0
        node_author.status = "failed"

    # Publication year: requires title
    if _non_empty(info.title):
        claims_and_sources.append((
            f"The cookbook '{info.title.strip()}' was published in 2016.",
            sources if sources else None,
            node_pub_year,
            "Check the publication date/year shown on the cited pages. If multiple editions are present, it's acceptable if any official edition for this title is published in 2016."
        ))
    else:
        node_pub_year.score = 0.0
        node_pub_year.status = "failed"

    # Recipe count: requires title
    if _non_empty(info.title):
        claims_and_sources.append((
            f"The cookbook '{info.title.strip()}' contains exactly 130 recipes.",
            sources if sources else None,
            node_recipe_count,
            "Verify that the page explicitly indicates 130 recipes (not 'about' or 'more than')."
        ))
    else:
        node_recipe_count.score = 0.0
        node_recipe_count.status = "failed"

    # Content theme: requires title
    if _non_empty(info.title):
        claims_and_sources.append((
            f"The cookbook '{info.title.strip()}' focuses on outdoor cooking and grilling adventures.",
            sources if sources else None,
            node_theme,
            "Look for descriptions indicating outdoor cooking, grilling, barbecue, or similar. Minor paraphrases are acceptable as long as the theme clearly centers on outdoor grilling adventures."
        ))
    else:
        node_theme.score = 0.0
        node_theme.status = "failed"

    # Page count: requires title
    if _non_empty(info.title):
        claims_and_sources.append((
            f"The cookbook '{info.title.strip()}' has 352 pages.",
            sources if sources else None,
            node_page_count,
            "Verify the page count. If multiple formats/editions show different counts, accept the one that explicitly lists 352 pages."
        ))
    else:
        node_page_count.score = 0.0
        node_page_count.status = "failed"

    # Author background: requires author
    if _non_empty(info.author):
        claims_and_sources.append((
            f"The author {info.author.strip()} is a television personality.",
            sources if sources else None,
            node_author_bg,
            "Pages like Wikipedia, official bios, or media sites stating the person is a TV personality/host are acceptable. Treat 'TV host' or 'television personality' as equivalent."
        ))
    else:
        node_author_bg.score = 0.0
        node_author_bg.status = "failed"

    # Restaurant locations: requires author
    if _non_empty(info.author):
        claims_and_sources.append((
            f"The author {info.author.strip()} has restaurant locations in both California and Nevada.",
            sources if sources else None,
            node_restaurant_states,
            "Accept evidence from official restaurant pages, bios, or credible sources listing restaurant locations. Recognize state abbreviations (CA for California, NV for Nevada)."
        ))
    else:
        node_restaurant_states.score = 0.0
        node_restaurant_states.status = "failed"

    # Publisher: requires title
    if _non_empty(info.title):
        claims_and_sources.append((
            f"The cookbook '{info.title.strip()}' was published by William Morrow Cookbooks.",
            sources if sources else None,
            node_publisher,
            "Accept 'William Morrow Cookbooks' directly. If the page lists 'William Morrow' but clearly indicates the 'Cookbooks' imprint for this title, consider it equivalent."
        ))
    else:
        node_publisher.score = 0.0
        node_publisher.status = "failed"

    # Batch verify all prepared claims in parallel
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the cookbook identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root can be parallel; the critical gating is at the child node
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_cookbook_info(),
        template_class=CookbookExtraction,
        extraction_name="cookbook_identification"
    )

    # Record constraints as Ground Truth context (for transparency only)
    evaluator.add_ground_truth({
        "required_publication_year": "2016",
        "required_recipe_count": "130",
        "required_theme": "outdoor cooking and grilling adventures",
        "required_page_count": "352",
        "required_author_background": "television personality",
        "required_restaurant_states": ["California", "Nevada"],
        "required_publisher": "William Morrow Cookbooks",
        "deliverables": ["exact title", "author name"]
    }, gt_type="constraints")

    # Build verification subtree and run checks
    await build_and_verify_cookbook_nodes(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()