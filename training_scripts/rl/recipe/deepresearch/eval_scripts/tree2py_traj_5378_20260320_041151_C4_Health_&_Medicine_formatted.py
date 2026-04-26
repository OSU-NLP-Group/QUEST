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
TASK_ID = "wsj_dga_2025_2030_article_check"
TASK_DESCRIPTION = (
    "Identify a news article published by the Wall Street Journal on January 7, 2026, "
    "that reports on the 2025-2030 Dietary Guidelines for Americans. The article must "
    "discuss the new protein intake recommendation of 1.2-1.6 grams per kilogram of "
    "body weight per day for adults, mention that the saturated fat limit is maintained "
    "at 10% of total daily calories, describe the change from MyPlate to a pyramid-style "
    "visual representation, include information about the recommendation to consume full-fat "
    "dairy (or the departure from previous low-fat dairy advice), and address the guidance on "
    "added sugars or the recommendation to avoid/eliminate added sugars. Provide the complete "
    "title of the article and its URL."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ArticleCandidate(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    outlet: Optional[str] = None
    publication_date: Optional[str] = None


class ArticleListExtraction(BaseModel):
    articles: List[ArticleCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_articles() -> str:
    return """
    From the provided answer, extract all candidate news article references mentioned that could match
    the user's request. For each candidate, extract the following fields exactly as written in the answer:
    1) title: The complete article title (verbatim, if present).
    2) url: The article URL (must be an explicit URL in the answer; do not invent one).
    3) outlet: The publication name if mentioned (e.g., "The Wall Street Journal", "WSJ").
    4) publication_date: The publication date string if explicitly provided in the answer (verbatim).
    
    Return a JSON object with:
      {
        "articles": [
          {"title": ..., "url": ..., "outlet": ..., "publication_date": ...},
          ...
        ]
      }
    If a field is not present for a candidate, set it to null. Only include entries that have at least a title or a URL.
    Do not add or infer information not explicitly stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper to choose the main article to verify                                 #
# --------------------------------------------------------------------------- #
def choose_main_article(extraction: ArticleListExtraction) -> ArticleCandidate:
    # Prefer a WSJ link if available; otherwise fallback to the first with a URL; else the first entry; else empty
    if not extraction or not extraction.articles:
        return ArticleCandidate()

    # First try exact WSJ domain
    for art in extraction.articles:
        if art.url and "wsj.com" in art.url.lower():
            return art

    # Fallback: first with any URL
    for art in extraction.articles:
        if art.url:
            return art

    # Fallback: first entry
    return extraction.articles[0]


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
    # Initialize evaluator with a parallel root (the rubric is parallel across required checks)
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

    # Extract candidate articles referenced in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_articles(),
        template_class=ArticleListExtraction,
        extraction_name="article_candidates",
    )

    # Choose the main article (prefer WSJ link)
    main_article = choose_main_article(extracted)
    main_url = main_article.url
    main_title = main_article.title or ""
    main_outlet = main_article.outlet or ""
    main_date = main_article.publication_date or ""

    # Record selected target info for transparency
    evaluator.add_custom_info(
        info={
            "selected_url": main_url,
            "selected_title": main_title,
            "selected_outlet": main_outlet,
            "selected_publication_date_text_in_answer": main_date,
            "all_extracted_articles": [a.dict() for a in extracted.articles],
        },
        info_type="selection_summary",
    )

    # Common verification guidance appended to each check
    COMMON_ADD_INS = (
        "Use only the provided webpage as evidence. If the URL is missing, invalid, inaccessible, "
        "or not a Wall Street Journal article page, conclude the claim is NOT supported. "
        "Minor wording or formatting differences are acceptable (e.g., 'Jan. 7, 2026' vs 'January 7, 2026', "
        "'g/kg/day' vs 'grams per kilogram per day', etc.). If a paywall limits text, carefully review the visible "
        "headline/deck/byline/date blocks and any visible body text and screenshots."
    )

    # Build leaf nodes as specified in the rubric (all critical)
    node_outlet = evaluator.add_leaf(
        id="publication_outlet",
        desc="The article is published by the Wall Street Journal",
        parent=root,
        critical=True,
    )
    node_date = evaluator.add_leaf(
        id="publication_date",
        desc="The article was published on January 7, 2026",
        parent=root,
        critical=True,
    )
    node_protein = evaluator.add_leaf(
        id="protein_recommendation",
        desc="The article discusses or mentions the new protein intake recommendation of 1.2-1.6 grams per kilogram of body weight per day for adults",
        parent=root,
        critical=True,
    )
    node_satfat = evaluator.add_leaf(
        id="saturated_fat_limit",
        desc="The article mentions that the saturated fat limit is maintained at 10% of total daily calories",
        parent=root,
        critical=True,
    )
    node_visual = evaluator.add_leaf(
        id="visual_change",
        desc="The article discusses the change from MyPlate to a new pyramid structure or mentions the return of a pyramid-style visual representation",
        parent=root,
        critical=True,
    )
    node_dairy = evaluator.add_leaf(
        id="full_fat_dairy",
        desc="The article mentions the recommendation to consume full-fat dairy or the departure from previous low-fat dairy advice",
        parent=root,
        critical=True,
    )
    node_sugar = evaluator.add_leaf(
        id="added_sugar_guidance",
        desc="The article mentions the guidance on added sugars or the recommendation to avoid/eliminate added sugars",
        parent=root,
        critical=True,
    )

    # Build claims
    claim_outlet = "This webpage is a news article published by The Wall Street Journal (WSJ)."
    claim_date = "This article was published on January 7, 2026."
    claim_protein = (
        "The article mentions a new protein intake recommendation of approximately 1.2 to 1.6 grams per kilogram "
        "of body weight per day for adults (allow equivalents like 'g/kg/day' or 'grams per kilogram per day')."
    )
    claim_satfat = (
        "The article states that the saturated fat limit remains at 10% of total daily calories (i.e., unchanged at 10%)."
    )
    claim_visual = (
        "The article discusses that the visual guide is changing from MyPlate to a pyramid-style representation "
        "(i.e., a return to a food pyramid or similar pyramid graphic)."
    )
    claim_dairy = (
        "The article mentions a recommendation allowing or endorsing full-fat (whole-fat) dairy, or otherwise indicates "
        "a departure from previous low-fat-only dairy guidance."
    )
    claim_sugar = (
        "The article includes guidance on added sugars recommending to avoid or eliminate added sugars "
        "(e.g., 'no added sugars', 'avoid added sugars', or similar)."
    )

    # Prepare batch verifications
    claims_and_sources = [
        (claim_outlet, main_url, node_outlet, COMMON_ADD_INS + " Accept 'WSJ' branding as Wall Street Journal."),
        (
            claim_date,
            main_url,
            node_date,
            COMMON_ADD_INS
            + " Accept common date formats like 'Jan. 7, 2026', 'January 7, 2026', or 'Jan 7, 2026'. If the page shows a"
            " publication date or an updated date on Jan 7, 2026, consider it sufficient.",
        ),
        (
            claim_protein,
            main_url,
            node_protein,
            COMMON_ADD_INS
            + " Look for a range around 1.2–1.6 g/kg/day for adults; minor wording variations count if the numeric range is clear.",
        ),
        (
            claim_satfat,
            main_url,
            node_satfat,
            COMMON_ADD_INS
            + " Look for explicit statements that the saturated fat limit is 10% of daily calories and that it is maintained/unchanged.",
        ),
        (
            claim_visual,
            main_url,
            node_visual,
            COMMON_ADD_INS
            + " Look for references to replacing 'MyPlate' with a pyramid or a return of a pyramid-style visual.",
        ),
        (
            claim_dairy,
            main_url,
            node_dairy,
            COMMON_ADD_INS
            + " Look for mentions of 'full-fat', 'whole milk/dairy', 'higher-fat dairy', or explicit contrasts with previous 'low-fat' advice.",
        ),
        (
            claim_sugar,
            main_url,
            node_sugar,
            COMMON_ADD_INS
            + " Look for explicit guidance to avoid/eliminate added sugars (wording like 'avoid', 'eliminate', 'no added sugars' counts).",
        ),
    ]

    # Run all verifications in parallel
    await evaluator.batch_verify(claims_and_sources)

    # Return structured summary
    return evaluator.get_summary()