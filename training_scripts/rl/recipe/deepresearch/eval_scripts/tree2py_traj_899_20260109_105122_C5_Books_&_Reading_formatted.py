import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "debut_novel_2023_prize"
TASK_DESCRIPTION = (
    "I'm looking for a debut novel published in the United States in 2023 that won a major first novel prize. "
    "This prize was announced in early December 2023 and carries a $15,000 award for the winner. The novel was "
    "published by an independent press that was established in 2020 and is distributed by Penguin Random House. "
    "The publication date of the novel was in the spring of 2023, specifically in April. The award ceremony took "
    "place at the Annual Awards Benefit in December 2023. What is the title of this debut novel, who is its author, "
    "what is the name of the publisher, and on what specific date was the award announced?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SourceBundle(BaseModel):
    """Categorized URLs explicitly cited in the answer."""
    book_page_urls: List[str] = Field(default_factory=list)
    publisher_about_urls: List[str] = Field(default_factory=list)
    distribution_urls: List[str] = Field(default_factory=list)
    prize_announcement_urls: List[str] = Field(default_factory=list)
    award_ceremony_urls: List[str] = Field(default_factory=list)
    general_urls: List[str] = Field(default_factory=list)


class NovelInfoExtraction(BaseModel):
    """Structured fields requested and constraint-related attributes extracted from the answer."""
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None  # keep as free-text to accommodate various formats
    country_of_publication: Optional[str] = None

    prize_name: Optional[str] = None
    prize_announcement_date: Optional[str] = None  # specific calendar date requested
    prize_amount: Optional[str] = None

    award_ceremony_name: Optional[str] = None
    award_ceremony_month_year: Optional[str] = None  # e.g., "December 2023"

    publisher_established_year: Optional[str] = None
    publisher_is_independent: Optional[str] = None  # e.g., "yes"/"no" or textual mention
    publisher_distributor: Optional[str] = None

    sources: SourceBundle = Field(default_factory=SourceBundle)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_novel_info() -> str:
    return """
    Extract from the answer the requested fields and all constraint-relevant attributes for the identified debut novel and its award.

    Required fields to extract (return null if missing):
    - title: The title of the debut novel.
    - author: The author of the debut novel.
    - publisher: The name of the novel's publisher.
    - prize_announcement_date: The specific calendar date the prize/winner announcement was made (e.g., "December 7, 2023").

    Additional attributes to help verification (return null if missing):
    - publication_date: The novel's publication date (free text; month/day/year if available; otherwise month/year).
    - country_of_publication: Country where it was published (e.g., "United States").
    - prize_name: The name of the prize (e.g., "The Center for Fiction First Novel Prize").
    - prize_amount: The prize amount for the winner (e.g., "$15,000").
    - award_ceremony_name: The award ceremony's name (e.g., "Annual Awards Benefit").
    - award_ceremony_month_year: The month and year of the ceremony (e.g., "December 2023").
    - publisher_established_year: The year the publisher/press was established (e.g., "2020").
    - publisher_is_independent: Does the answer state or imply the press is independent? Use "yes", "no", or a short phrase directly quoted.
    - publisher_distributor: The distributor name if mentioned (e.g., "Penguin Random House").

    Also extract URLs explicitly cited in the answer for later verification (categorize appropriately):
    - sources.book_page_urls: URLs that are book pages (publisher page for the book, retailer listing, official listing).
    - sources.publisher_about_urls: URLs that describe the publisher (about page, press info).
    - sources.distribution_urls: URLs explicitly showing the publisher’s distribution relationship (e.g., PRH distribution page).
    - sources.prize_announcement_urls: URLs announcing the prize winner or press releases/news posts for the prize.
    - sources.award_ceremony_urls: URLs describing the award ceremony (e.g., Annual Awards Benefit page).
    - sources.general_urls: Any other URL in the answer relevant to this task.

    IMPORTANT:
    - Only extract information explicitly present in the answer text.
    - For URLs, extract the actual URLs as they appear (markdown links are acceptable; extract the URL).
    - Do not invent data; return null if the answer did not provide it.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(val: Optional[str]) -> bool:
    return bool(val and isinstance(val, str) and val.strip() != "")


def _combine_sources(s: SourceBundle, keys: List[str]) -> List[str]:
    out: List[str] = []
    for k in keys:
        lst = getattr(s, k, [])
        if isinstance(lst, list):
            out.extend(lst)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in out:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
def build_required_fields_nodes(evaluator: Evaluator, parent_node, info: NovelInfoExtraction) -> None:
    """
    Build the 'required_answer_fields' subtree with critical existence checks.
    """
    required_node = evaluator.add_parallel(
        id="required_answer_fields",
        desc="Answer includes all fields asked for in the question.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(info.title),
        id="novel_title",
        desc="Provide the title of the debut novel.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty_str(info.author),
        id="author_name",
        desc="Provide the author of the debut novel.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty_str(info.publisher),
        id="publisher_name",
        desc="Provide the name of the novel's publisher.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty_str(info.prize_announcement_date),
        id="award_announcement_date",
        desc="Provide the specific calendar date the prize was announced.",
        parent=required_node,
        critical=True
    )


async def build_constraints_nodes(evaluator: Evaluator, parent_node, info: NovelInfoExtraction) -> None:
    """
    Build the 'constraint_verification' subtree and run URL-backed verifications.
    """
    constraints_node = evaluator.add_parallel(
        id="constraint_verification",
        desc="The identified novel/publisher/prize satisfy all stated constraints from the question/constraints list.",
        parent=parent_node,
        critical=True
    )

    # Helpful strings
    t = info.title or "the novel"
    a = info.author or "the author"
    p = info.publisher or "the publisher"
    pub_date_txt = info.publication_date or ""
    prize_name_txt = info.prize_name or "the prize"
    ann_date_txt = info.prize_announcement_date or ""
    ceremony_name_txt = info.award_ceremony_name or "Annual Awards Benefit"
    ceremony_my_txt = info.award_ceremony_month_year or "December 2023"

    # Source groups
    s_book = _combine_sources(info.sources, ["book_page_urls", "general_urls"])
    s_pub = _combine_sources(info.sources, ["publisher_about_urls", "general_urls"])
    s_dist = _combine_sources(info.sources, ["distribution_urls", "general_urls"])
    s_prize = _combine_sources(info.sources, ["prize_announcement_urls", "general_urls"])
    s_ceremony = _combine_sources(info.sources, ["award_ceremony_urls", "general_urls"])
    s_all = _combine_sources(info.sources, [
        "book_page_urls",
        "publisher_about_urls",
        "distribution_urls",
        "prize_announcement_urls",
        "award_ceremony_urls",
        "general_urls"
    ])

    # 1) Debut novel
    node_debut = evaluator.add_leaf(
        id="debut_novel",
        desc="Verify the book is a debut novel (the author's first novel).",
        parent=constraints_node,
        critical=True
    )
    claim_debut = f"'{t}' by {a} is the author's debut (first) novel."
    await evaluator.verify(
        claim=claim_debut,
        node=node_debut,
        sources=s_book or s_all,
        additional_instruction="Look for phrases like 'debut novel', 'first novel', or equivalent statements explicitly linked to this book/author."
    )

    # 2) Published in the United States in 2023
    node_us_2023 = evaluator.add_leaf(
        id="published_in_us_2023",
        desc="Verify the book was published in the United States in 2023.",
        parent=constraints_node,
        critical=True
    )
    claim_us_2023 = f"'{t}' was published in the United States in 2023."
    await evaluator.verify(
        claim=claim_us_2023,
        node=node_us_2023,
        sources=s_book or s_pub or s_all,
        additional_instruction=(
            "Accept if evidence indicates the publication took place in the U.S. (e.g., U.S. publisher) and the release year is 2023."
        )
    )

    # 3) Publication date in April 2023
    node_april = evaluator.add_leaf(
        id="publication_date_april_2023",
        desc="Verify the book's publication date is in April 2023.",
        parent=constraints_node,
        critical=True
    )
    claim_april = f"'{t}' was published in April 2023."
    await evaluator.verify(
        claim=claim_april,
        node=node_april,
        sources=s_book or s_all,
        additional_instruction="Check the publication/release date on book pages or official listings; accept if the date clearly falls within April 2023."
    )

    # 4) Publisher is independent press established in 2020
    node_independent_2020 = evaluator.add_leaf(
        id="publisher_independent_established_2020",
        desc="Verify the publisher is an independent press established in 2020.",
        parent=constraints_node,
        critical=True
    )
    claim_independent_2020 = f"{p} is an independent press established in 2020."
    await evaluator.verify(
        claim=claim_independent_2020,
        node=node_independent_2020,
        sources=s_pub or s_all,
        additional_instruction="Look for the publisher's 'About' page or credible sources explicitly stating they are independent and founded/established in 2020."
    )

    # 5) Publisher distributed by Penguin Random House
    node_prh = evaluator.add_leaf(
        id="publisher_distributed_by_prh",
        desc="Verify the publisher is distributed by Penguin Random House.",
        parent=constraints_node,
        critical=True
    )
    claim_prh = f"{p} is distributed by Penguin Random House."
    await evaluator.verify(
        claim=claim_prh,
        node=node_prh,
        sources=s_dist or s_pub or s_all,
        additional_instruction="Seek explicit distribution statements; accept equivalent phrasing (e.g., 'distribution through PRH', 'PRH handles distribution')."
    )

    # 6) Novel won a first/debut novel prize (is the prize winner)
    node_won = evaluator.add_leaf(
        id="novel_won_first_novel_prize",
        desc="Verify the novel won a first/debut novel prize (i.e., it is the prize winner).",
        parent=constraints_node,
        critical=True
    )
    claim_won = f"'{t}' won the {prize_name_txt} (a first/debut novel prize)."
    await evaluator.verify(
        claim=claim_won,
        node=node_won,
        sources=s_prize or s_all,
        additional_instruction="Confirm that the book is explicitly named as the winner of the specified prize. Do not accept shortlist or finalist status."
    )

    # 7) Prize is specifically for debut/first novels
    node_debut_prize = evaluator.add_leaf(
        id="prize_is_for_debut_novels",
        desc="Verify the award is specifically for debut/first novels.",
        parent=constraints_node,
        critical=True
    )
    claim_debut_prize = f"The {prize_name_txt} is an award specifically for debut/first novels."
    await evaluator.verify(
        claim=claim_debut_prize,
        node=node_debut_prize,
        sources=s_prize or s_all,
        additional_instruction="Check the prize description; accept language like 'First Novel Prize', 'debut novel award', or equivalent."
    )

    # 8) Prize announced in early December 2023
    node_early_dec = evaluator.add_leaf(
        id="prize_announced_early_dec_2023",
        desc="Verify the prize was announced in early December 2023.",
        parent=constraints_node,
        critical=True
    )
    claim_early_dec = f"The winner of the {prize_name_txt} was announced in early December 2023."
    await evaluator.verify(
        claim=claim_early_dec,
        node=node_early_dec,
        sources=s_prize or s_all,
        additional_instruction="Treat 'early December' as approximately December 1–10, 2023; accept ~Dec 1–12 if sources clearly frame it as 'early December'."
    )

    # 9) Prize amount is $15,000
    node_15k = evaluator.add_leaf(
        id="prize_amount_15000",
        desc="Verify the prize amount for the winner is $15,000.",
        parent=constraints_node,
        critical=True
    )
    claim_15k = f"The prize amount for the winner is $15,000."
    await evaluator.verify(
        claim=claim_15k,
        node=node_15k,
        sources=s_prize or s_all,
        additional_instruction="Allow numeric variants like '$15,000', '$15k', or 'fifteen thousand dollars'."
    )

    # 10) Award ceremony took place at Annual Awards Benefit in December 2023
    node_benefit = evaluator.add_leaf(
        id="award_ceremony_annual_awards_benefit_dec_2023",
        desc="Verify the award ceremony took place at an Annual Awards Benefit in December 2023.",
        parent=constraints_node,
        critical=True
    )
    claim_benefit = f"The award ceremony took place at the {ceremony_name_txt} in {ceremony_my_txt}."
    await evaluator.verify(
        claim=claim_benefit,
        node=node_benefit,
        sources=s_ceremony or s_prize or s_all,
        additional_instruction="Accept wording variants (e.g., 'Annual Awards Benefit', 'Awards Benefit Gala') and confirm the month/year is December 2023."
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for the debut novel prize task.
    """
    # Initialize evaluator (root is critical parallel as per rubric)
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
        default_model=model
    )
    # Make root critical to match rubric and enforce pass-all behavior
    root.critical = True

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_novel_info(),
        template_class=NovelInfoExtraction,
        extraction_name="novel_info_extraction"
    )

    # Add a summary of constraints as Ground Truth context (for transparency only)
    evaluator.add_ground_truth({
        "constraints": {
            "debut_novel": True,
            "published_in_us_2023": True,
            "publication_month": "April 2023",
            "publisher_independent_established_2020": True,
            "publisher_distributed_by_prh": True,
            "won_first_novel_prize": True,
            "prize_is_for_debut_novels": True,
            "prize_announced_early_dec_2023": True,
            "prize_amount": "$15,000",
            "award_ceremony_at_annual_awards_benefit_dec_2023": True
        }
    }, gt_type="constraints_summary")

    # Build required fields sub-tree first
    build_required_fields_nodes(evaluator, root, extracted_info)

    # Then build constraints subtree and run verifications
    await build_constraints_nodes(evaluator, root, extracted_info)

    # Return standardized summary with verification tree and score
    return evaluator.get_summary()