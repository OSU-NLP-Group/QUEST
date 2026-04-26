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
TASK_ID = "us_env_policy_article"
TASK_DESCRIPTION = (
    "Find a news article published by a major US news organization about recent US environmental or climate policy developments. "
    "The article must meet professional journalism standards and provide the article's URL, headline, author(s), publication date, and publication source."
)

EVALUATION_DATE_STR = "2026-01-09"
THIRTY_DAY_THRESHOLD_STR = "2025-12-10"

RECOGNIZED_MAJOR_OUTLETS = [
    "The New York Times",
    "The Washington Post",
    "The Wall Street Journal",
    "Los Angeles Times",
    "USA Today",
    "Associated Press",
    "AP News",
    "Reuters",
    "NPR",
    "CNN",
    "BBC",
    "ProPublica",
    "Bloomberg",
    "Politico",
    "ABC News",
    "CBS News",
    "Fox News",
    "The Guardian",
    "Guardian US",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArticleExtraction(BaseModel):
    """Extracted fields that the user was asked to provide."""
    article_url: Optional[str] = None
    headline: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    publication_date: Optional[str] = None
    publication_source: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_article_info() -> str:
    return (
        "From the answer, extract the news article details exactly as provided. Return the following fields:\n"
        "1. article_url: The URL to the article's webpage (must be a valid URL string if present).\n"
        "2. headline: The clear, descriptive headline of the article.\n"
        "3. authors: A list of author names shown in the byline.\n"
        "4. publication_date: The publication date string as shown (do not reformat; keep as-is).\n"
        "5. publication_source: The name of the publishing news organization (e.g., Reuters, AP News, The New York Times).\n"
        "If any field is missing in the answer, set it to null (or an empty list for authors). Do not invent information."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _join_authors(authors: List[str]) -> str:
    return ", ".join(a for a in authors if a and a.strip()) if authors else "None"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_article_tree(
    evaluator: Evaluator,
    root: Any,
    info: ArticleExtraction,
) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """
    # Create top-level node (parallel aggregation; non-critical for partial credit)
    top_node = evaluator.add_parallel(
        id="article_identification",
        desc="Find a news article about recent US environmental or climate policy that meets professional journalism standards",
        parent=root,
        critical=False,
    )

    # Ensure Online_Accessible is created and verified first, so we can gate other checks on it
    online_accessible_node = evaluator.add_leaf(
        id="online_accessible",
        desc="The article must be accessible online through a valid, working URL",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided URL '{info.article_url}' is a valid, working webpage that loads the article.",
        node=online_accessible_node,
        sources=info.article_url,
        additional_instruction=(
            "If the URL is missing or invalid, mark incorrect. This verification must rely on the webpage content."
        ),
    )

    # Major news source
    major_source_node = evaluator.add_leaf(
        id="major_news_source",
        desc="The article must be published by a recognized major US news organization",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This article is published by '{info.publication_source}', which is a recognized major US news organization."
        ),
        node=major_source_node,
        sources=info.article_url,
        additional_instruction=(
            "Check the publisher shown on the article page. Recognized outlets include: "
            + "; ".join(RECOGNIZED_MAJOR_OUTLETS)
            + ". If the outlet is an equivalent tier-1 national news source, consider it recognized."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Recent publication (on or after threshold date)
    recent_pub_node = evaluator.add_leaf(
        id="recent_publication",
        desc=f"The article must have been published on or after {THIRTY_DAY_THRESHOLD_STR}",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article was published on or after {THIRTY_DAY_THRESHOLD_STR}.",
        node=recent_pub_node,
        sources=info.article_url,
        additional_instruction=(
            f"Use the publication date displayed on the page. The evaluation date is {EVALUATION_DATE_STR}, "
            f"so 'within the past 30 days' means on or after {THIRTY_DAY_THRESHOLD_STR}. "
            "If only an 'Updated' date is shown but the original publish date is older than the threshold, do not pass."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Headline present (and matches/aligns with provided headline)
    headline_node = evaluator.add_leaf(
        id="headline_present",
        desc="The article must have a clear, descriptive headline that summarizes the main topic",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The article page displays a clear headline that matches or is equivalent to '{info.headline}'."
        ),
        node=headline_node,
        sources=info.article_url,
        additional_instruction=(
            "Allow minor variations like punctuation, capitalization, or short suffix/prefix additions. "
            "The headline should be clearly visible near the top of the article page."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Byline present (authors identified)
    byline_node = evaluator.add_leaf(
        id="byline_present",
        desc="The article must include a byline that identifies the author(s)",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article includes a byline identifying the authors: {_join_authors(info.authors)}.",
        node=byline_node,
        sources=info.article_url,
        additional_instruction=(
            "Verify that author names are present in a byline on the page. "
            "Minor variations in name formatting are acceptable (e.g., middle initials, suffixes)."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Dateline present (reporting location)
    dateline_node = evaluator.add_leaf(
        id="dateline_present",
        desc="The article must include a dateline indicating the reporting location",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The article page includes a dateline indicating the reporting location (e.g., city/state/country).",
        node=dateline_node,
        sources=info.article_url,
        additional_instruction=(
            "A dateline typically appears at the start of the story or near the byline, showing a location like 'WASHINGTON' or 'SACRAMENTO, Calif.' "
            "If no location indicator is present in the article body or standard dateline area, do not pass."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Minimum word count (500 words)
    word_count_node = evaluator.add_leaf(
        id="minimum_word_count",
        desc="The article must contain at least 500 words of substantive content",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The article contains at least 500 words of substantive content.",
        node=word_count_node,
        sources=info.article_url,
        additional_instruction=(
            "Focus on the article body content only (exclude navigation, unrelated widgets, and boilerplate). "
            "A reasonable approximation is acceptable."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Source attribution
    attribution_node = evaluator.add_leaf(
        id="source_attribution",
        desc="The article must properly attribute information to at least one identifiable source",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The article attributes information to at least one identifiable source (person, organization, or document).",
        node=attribution_node,
        sources=info.article_url,
        additional_instruction=(
            "Examples include named individuals, official documents, court filings, agency reports, or organizational statements. "
            "Generic phrases like 'experts say' without identifiable sources should not pass."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Direct quote present
    quote_node = evaluator.add_leaf(
        id="direct_quote_present",
        desc="The article must include at least one direct quotation from a relevant source",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The article includes at least one direct quotation from a relevant source.",
        node=quote_node,
        sources=info.article_url,
        additional_instruction=(
            "Look for text enclosed in quotation marks or displayed as a block quote attributed to a person or official statement."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Five W's answered
    fivews_node = evaluator.add_leaf(
        id="five_ws_answered",
        desc="The article must answer who, what, when, where, and why",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The article answers who, what, when, where, and why.",
        node=fivews_node,
        sources=info.article_url,
        additional_instruction=(
            "Assess whether the fundamental journalism questions are addressed within the article body and lead. "
            "Minor omissions may still fail if one of the W's is not answered."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Lead paragraph present
    lead_node = evaluator.add_leaf(
        id="lead_paragraph",
        desc="The article must have a clear lead paragraph delivering key facts",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The article contains a clear lead paragraph that delivers the key facts of the story.",
        node=lead_node,
        sources=info.article_url,
        additional_instruction=(
            "The lead should summarize the essential facts and set context at the top of the article body."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # US-related topic focus (environment/climate policy)
    us_topic_node = evaluator.add_leaf(
        id="us_related_topic",
        desc="The article must be about US environmental or climate policy (federal, state, or significant local level)",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The article focuses on US environmental or climate policy at the federal, state, or significant local level.",
        node=us_topic_node,
        sources=info.article_url,
        additional_instruction=(
            "Look for references to US jurisdictions, agencies (EPA, DOE, Interior), Congress, White House, US states, or local governments. "
            "If primarily about non-US policy, do not pass."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Publication date visible
    pub_date_visible_node = evaluator.add_leaf(
        id="publication_date_visible",
        desc="The article must clearly display its publication date on the webpage",
        parent=top_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The article clearly displays its publication date on the webpage.",
        node=pub_date_visible_node,
        sources=info.article_url,
        additional_instruction=(
            "The publication date should be clearly visible near the headline or byline. "
            "If only an update timestamp exists without an original publish date, consider the outlet's conventions and visibility."
        ),
        extra_prerequisites=[online_accessible_node],
    )

    # Multimedia element (non-critical)
    multimedia_node = evaluator.add_leaf(
        id="multimedia_element",
        desc="The article should include at least one multimedia element (photograph, video, infographic, or interactive graphic)",
        parent=top_node,
        critical=False,
    )
    await evaluator.verify(
        claim="The article includes at least one multimedia element (photo, video, infographic, or interactive graphic).",
        node=multimedia_node,
        sources=info.article_url,
        additional_instruction=(
            "Look for embedded images, videos, infographics, or interactive graphics within the article page."
        ),
        extra_prerequisites=[online_accessible_node],
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
    Evaluate an answer for the US environmental/climate policy article task.
    """
    # Initialize evaluator
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

    # Extract article fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_article_info(),
        template_class=ArticleExtraction,
        extraction_name="article_fields",
    )

    # Record criteria info for transparency
    evaluator.add_custom_info(
        info={
            "evaluation_date": EVALUATION_DATE_STR,
            "threshold_date": THIRTY_DAY_THRESHOLD_STR,
            "recognized_outlets": RECOGNIZED_MAJOR_OUTLETS,
        },
        info_type="criteria",
        info_name="journalism_criteria",
    )

    # Build tree and perform verifications
    await build_and_verify_article_tree(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()