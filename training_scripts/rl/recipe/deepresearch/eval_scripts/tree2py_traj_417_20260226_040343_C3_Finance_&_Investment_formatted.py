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
TASK_ID = "asset_firm_1975_pa"
TASK_DESCRIPTION = (
    "Identify the asset management firm that was founded in 1975 and is headquartered in Pennsylvania. "
    "Who founded this firm? What was the complete title of the Princeton University senior thesis that this founder wrote? "
    "What Fortune magazine article inspired this thesis, and in what month and year was it published? "
    "Provide the exact title of that Fortune magazine article."
)

# Optional ground truth information for reference in the summary (not used for scoring directly)
GROUND_TRUTH = {
    "expected_firm": "The Vanguard Group",
    "expected_founder": "John C. Bogle",
    "expected_thesis_title": "The Economic Role of the Investment Company",
    "expected_thesis_year": "1951",
    "expected_fortune_article_title": "Big Money in Boston",
    "expected_fortune_article_month": "December",
    "expected_fortune_article_year": "1949",
    "expected_article_subjects": ["Massachusetts Investors Trust", "mutual fund industry"]
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FirmInfo(BaseModel):
    name: Optional[str] = None
    headquarters: Optional[str] = None  # e.g., "Malvern, Pennsylvania" or just "Pennsylvania"
    founded_year: Optional[str] = None  # keep as string to allow variants like "1975" or "founded in 1975"
    sources: List[str] = Field(default_factory=list)  # URLs explicitly cited for the firm


class FounderInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # URLs that support the founder-firm relationship


class ThesisInfo(BaseModel):
    university: Optional[str] = None  # expect "Princeton University"
    title: Optional[str] = None
    completion_year: Optional[str] = None  # expect "1951"
    sources: List[str] = Field(default_factory=list)  # URLs that support thesis details


class FortuneArticleInfo(BaseModel):
    title: Optional[str] = None  # expect "Big Money in Boston"
    publication_month: Optional[str] = None  # expect "December" (allow variants like "Dec")
    publication_year: Optional[str] = None  # expect "1949"
    subjects: List[str] = Field(default_factory=list)  # e.g., ["Massachusetts Investors Trust", "mutual fund industry"]
    sources: List[str] = Field(default_factory=list)  # URLs that support article details


class AnswerExtraction(BaseModel):
    firm: Optional[FirmInfo] = None
    founder: Optional[FounderInfo] = None
    thesis: Optional[ThesisInfo] = None
    article: Optional[FortuneArticleInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extraction() -> str:
    return """
    Extract structured information from the answer about the following four entities. If multiple candidates are mentioned, select the entity that matches all the constraints in the task (founded in 1975, headquartered in Pennsylvania; the thesis must be a Princeton University senior thesis; the Fortune article must have an exact title and a publication month/year).

    1) firm:
       - name: The name of the asset management firm explicitly named in the answer.
       - headquarters: The headquarters location string as stated (city/state or state only).
       - founded_year: The founding year string as stated in the answer (e.g., "1975", "founded in 1975").
       - sources: All URLs explicitly cited that support facts about the firm (home page, Wikipedia, official pages, reputable articles). Extract only URLs present in the answer.

    2) founder:
       - name: The founder's name as stated in the answer.
       - sources: All URLs explicitly cited that support the founder-firm relationship or biographical details.

    3) thesis:
       - university: The university for the senior thesis (expect "Princeton University").
       - title: The complete title of the thesis, exactly as written in the answer.
       - completion_year: The completion year as stated (expect "1951").
       - sources: All URLs explicitly cited that support the thesis details (Princeton library pages, reputable biographies, etc.).

    4) article:
       - title: The exact title of the Fortune magazine article (expect "Big Money in Boston").
       - publication_month: The publication month as stated (e.g., "December", "Dec").
       - publication_year: The publication year as stated (expect "1949").
       - subjects: Any subjects the answer claims the article featured (e.g., "Massachusetts Investors Trust", "mutual fund industry").
       - sources: All URLs explicitly cited that support the article details.

    IMPORTANT:
    - Extract only information explicitly stated in the answer text. Do not invent or infer any new facts.
    - Extract only valid URLs that appear in the answer (including markdown links).
    - If any field is not mentioned, return null for that field (or an empty list for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*source_lists: List[str]) -> List[str]:
    """Merge multiple URL lists into a unique, order-preserving list."""
    seen = set()
    merged: List[str] = []
    for sl in source_lists:
        for url in sl or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _safe_name(name: Optional[str], fallback: str) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else fallback


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_identify_firm_subtree(evaluator: Evaluator,
                                      parent_node,
                                      extraction: AnswerExtraction) -> None:
    """Build and verify the 'Identify_Firm' subtree."""
    firm = extraction.firm or FirmInfo()
    firm_name = _safe_name(firm.name, "the named firm")
    firm_sources = firm.sources

    identify_node = evaluator.add_parallel(
        id="Identify_Firm",
        desc="Identify an asset management firm that matches the founding-year and headquarters-location requirements.",
        parent=parent_node,
        critical=True
    )

    # Firm_Name_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(firm.name and firm.name.strip()),
        id="Firm_Name_Provided",
        desc="Answer explicitly names an asset management firm.",
        parent=identify_node,
        critical=True
    )

    # Firm_Is_Asset_Management_Firm
    node_is_asset = evaluator.add_leaf(
        id="Firm_Is_Asset_Management_Firm",
        desc="The named entity is in fact an asset management firm.",
        parent=identify_node,
        critical=True
    )
    claim_is_asset = f"The company {firm_name} is an asset management firm (e.g., investment management company or mutual fund company)."
    await evaluator.verify(
        claim=claim_is_asset,
        node=node_is_asset,
        sources=firm_sources,
        additional_instruction=(
            "Confirm via the cited pages that the entity operates as an asset/investment management company, "
            "including mutual fund management. Allow synonyms such as 'investment company', 'mutual fund company', or 'investment management firm'."
        )
    )

    # Firm_Founded_1975
    node_founded_1975 = evaluator.add_leaf(
        id="Firm_Founded_1975",
        desc="The named firm was founded in 1975.",
        parent=identify_node,
        critical=True
    )
    claim_founded_1975 = f"The company {firm_name} was founded in 1975."
    await evaluator.verify(
        claim=claim_founded_1975,
        node=node_founded_1975,
        sources=firm_sources,
        additional_instruction=(
            "Verify language such as 'founded in 1975', 'established in 1975', or equivalent phrasing on the cited pages."
        )
    )

    # Firm_HQ_Pennsylvania
    node_hq_pa = evaluator.add_leaf(
        id="Firm_HQ_Pennsylvania",
        desc="The named firm is headquartered in Pennsylvania.",
        parent=identify_node,
        critical=True
    )
    claim_hq_pa = f"The company {firm_name} is headquartered in Pennsylvania."
    await evaluator.verify(
        claim=claim_hq_pa,
        node=node_hq_pa,
        sources=firm_sources,
        additional_instruction=(
            "Confirm that the headquarters is in Pennsylvania (e.g., 'Malvern, Pennsylvania', 'Valley Forge, PA'). "
            "Minor formatting variations (e.g., 'PA') are acceptable."
        )
    )


async def build_provide_founder_subtree(evaluator: Evaluator,
                                        parent_node,
                                        extraction: AnswerExtraction) -> None:
    """Build and verify the 'Provide_Founder' subtree."""
    firm = extraction.firm or FirmInfo()
    founder = extraction.founder or FounderInfo()

    firm_name = _safe_name(firm.name, "the named firm")
    founder_name = _safe_name(founder.name, "the named founder")

    combined_sources = _merge_sources(founder.sources, firm.sources)

    provide_founder_node = evaluator.add_parallel(
        id="Provide_Founder",
        desc="Provide the founder of the identified firm.",
        parent=parent_node,
        critical=True
    )

    # Founder_Name_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(founder.name and founder.name.strip()),
        id="Founder_Name_Provided",
        desc="Answer provides the founder's name.",
        parent=provide_founder_node,
        critical=True
    )

    # Founder_Matches_Firm
    node_founder_matches = evaluator.add_leaf(
        id="Founder_Matches_Firm",
        desc="The named person is in fact the founder of the identified firm.",
        parent=provide_founder_node,
        critical=True
    )
    claim_founder_matches = f"{founder_name} is the founder of {firm_name}."
    await evaluator.verify(
        claim=claim_founder_matches,
        node=node_founder_matches,
        sources=combined_sources,
        additional_instruction=(
            "Confirm the founder relationship from the cited pages. Accept wording like 'founded by', 'founder', "
            "or 'established by'. If multiple founders are claimed, ensure the named person is indeed a founder."
        )
    )


async def build_provide_thesis_subtree(evaluator: Evaluator,
                                       parent_node,
                                       extraction: AnswerExtraction) -> None:
    """Build and verify the 'Provide_Thesis_Details' subtree."""
    founder = extraction.founder or FounderInfo()
    thesis = extraction.thesis or ThesisInfo()

    founder_name = _safe_name(founder.name, "the named founder")
    thesis_sources = thesis.sources

    provide_thesis_node = evaluator.add_parallel(
        id="Provide_Thesis_Details",
        desc="Provide the founder's Princeton University senior thesis details, including the required completion year and required complete title.",
        parent=parent_node,
        critical=True
    )

    # Founder_Wrote_Princeton_Senior_Thesis
    node_thesis_wrote = evaluator.add_leaf(
        id="Founder_Wrote_Princeton_Senior_Thesis",
        desc="The founder wrote a senior thesis at Princeton University.",
        parent=provide_thesis_node,
        critical=True
    )
    claim_thesis_wrote = f"{founder_name} wrote a senior thesis at Princeton University."
    await evaluator.verify(
        claim=claim_thesis_wrote,
        node=node_thesis_wrote,
        sources=thesis_sources,
        additional_instruction=(
            "Confirm that the thesis is a Princeton University senior/undergraduate thesis (often called AB senior thesis). "
            "Minor wording variants like 'undergraduate thesis' or 'senior thesis' are acceptable."
        )
    )

    # Thesis_Completed_1951
    node_thesis_year = evaluator.add_leaf(
        id="Thesis_Completed_1951",
        desc="The thesis was completed in 1951.",
        parent=provide_thesis_node,
        critical=True
    )
    claim_thesis_year = f"{founder_name}'s Princeton senior thesis was completed in 1951."
    await evaluator.verify(
        claim=claim_thesis_year,
        node=node_thesis_year,
        sources=thesis_sources,
        additional_instruction=(
            "Verify the completion year '1951' on the cited pages. Reasonable date formatting variants are acceptable."
        )
    )

    # Thesis_Title_Matches_Constraint
    node_thesis_title = evaluator.add_leaf(
        id="Thesis_Title_Matches_Constraint",
        desc="The thesis complete title matches the required title: 'The Economic Role of the Investment Company'.",
        parent=provide_thesis_node,
        critical=True
    )
    claim_thesis_title = "The complete title of the Princeton senior thesis is 'The Economic Role of the Investment Company'."
    await evaluator.verify(
        claim=claim_thesis_title,
        node=node_thesis_title,
        sources=thesis_sources,
        additional_instruction=(
            "Confirm that the exact phrase 'The Economic Role of the Investment Company' appears as the thesis title "
            "on the cited pages. Minor punctuation or casing variants are acceptable only if clearly the same title."
        )
    )


async def build_provide_fortune_subtree(evaluator: Evaluator,
                                        parent_node,
                                        extraction: AnswerExtraction) -> None:
    """Build and verify the 'Provide_Fortune_Article_Details' subtree."""
    founder = extraction.founder or FounderInfo()
    thesis = extraction.thesis or ThesisInfo()
    article = extraction.article or FortuneArticleInfo()

    founder_name = _safe_name(founder.name, "the named founder")
    article_sources = article.sources
    combined_sources = _merge_sources(article_sources, thesis.sources)

    provide_article_node = evaluator.add_parallel(
        id="Provide_Fortune_Article_Details",
        desc="Provide the Fortune magazine article that inspired the thesis, including required exact title, required publication month/year, and required subject matter.",
        parent=parent_node,
        critical=True
    )

    # Thesis_Inspired_By_Fortune_Article
    node_inspired = evaluator.add_leaf(
        id="Thesis_Inspired_By_Fortune_Article",
        desc="Answer states the thesis was inspired by a Fortune magazine article.",
        parent=provide_article_node,
        critical=True
    )
    claim_inspired = f"{founder_name}'s Princeton senior thesis was inspired by a Fortune magazine article."
    await evaluator.verify(
        claim=claim_inspired,
        node=node_inspired,
        sources=combined_sources,
        additional_instruction=(
            "Verify that the cited pages explicitly state the thesis was inspired by a Fortune magazine article. "
            "Paraphrases like 'inspired by a Fortune article' or 'prompted by an article in Fortune' are acceptable if clearly equivalent."
        )
    )

    # Fortune_Article_Title_Matches_Constraint
    node_article_title = evaluator.add_leaf(
        id="Fortune_Article_Title_Matches_Constraint",
        desc="The Fortune magazine article exact title matches the required title: 'Big Money in Boston'.",
        parent=provide_article_node,
        critical=True
    )
    claim_article_title = "The exact title of the Fortune magazine article is 'Big Money in Boston'."
    await evaluator.verify(
        claim=claim_article_title,
        node=node_article_title,
        sources=article_sources,
        additional_instruction=(
            "Confirm that the article's title is exactly 'Big Money in Boston' on the cited pages. "
            "Minor casing or punctuation variants are acceptable if clearly the same title."
        )
    )

    # Fortune_Article_Published_Dec_1949
    node_article_date = evaluator.add_leaf(
        id="Fortune_Article_Published_Dec_1949",
        desc="The Fortune magazine article publication month and year are December 1949.",
        parent=provide_article_node,
        critical=True
    )
    claim_article_date = "The Fortune magazine article 'Big Money in Boston' was published in December 1949."
    await evaluator.verify(
        claim=claim_article_date,
        node=node_article_date,
        sources=article_sources,
        additional_instruction=(
            "Confirm the publication month and year 'December 1949' (allow variants like 'Dec 1949')."
        )
    )

    # Article_Featured_MIT_And_Mutual_Fund_Industry
    node_article_subjects = evaluator.add_leaf(
        id="Article_Featured_MIT_And_Mutual_Fund_Industry",
        desc="The described Fortune article featured Massachusetts Investors Trust and the mutual fund industry.",
        parent=provide_article_node,
        critical=True
    )
    claim_article_subjects = (
        "The Fortune article 'Big Money in Boston' featured Massachusetts Investors Trust (MIT) and the mutual fund industry."
    )
    await evaluator.verify(
        claim=claim_article_subjects,
        node=node_article_subjects,
        sources=article_sources,
        additional_instruction=(
            "Verify that the article (or reliable summaries of it) explicitly mentions Massachusetts Investors Trust "
            "and discusses the mutual fund industry."
        )
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
    Evaluate the answer for the asset management firm founded in 1975 and headquartered in Pennsylvania,
    along with the founder, thesis details, and Fortune article details.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: non-critical; we will add a critical sequential child
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
    extraction = await evaluator.extract(
        prompt=prompt_extraction(),
        template_class=AnswerExtraction,
        extraction_name="structured_extraction"
    )

    # Add a critical sequential node representing the complete task (as per rubric)
    complete_task_node = evaluator.add_sequential(
        id="Complete_Task",
        desc=(
            "Identify an asset management firm founded in 1975 and headquartered in Pennsylvania, "
            "then provide its founder, the founder's Princeton senior thesis (with required title and completion year), "
            "and the Fortune magazine article (with required exact title and publication month/year) that inspired the thesis."
        ),
        parent=root,
        critical=True
    )

    # Build subtrees according to rubric order (sequential under Complete_Task)
    await build_identify_firm_subtree(evaluator, complete_task_node, extraction)
    await build_provide_founder_subtree(evaluator, complete_task_node, extraction)
    await build_provide_thesis_subtree(evaluator, complete_task_node, extraction)
    await build_provide_fortune_subtree(evaluator, complete_task_node, extraction)

    # Record optional ground truth info for transparency in summary
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="expected_facts")

    # Return structured summary
    return evaluator.get_summary()