import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pulitzer_nba_2024_publisher"
TASK_DESCRIPTION = """
What is the publisher of the novel that won both the Pulitzer Prize for Fiction and the National Book Award for Fiction in 2024?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookAndPublisherInfo(BaseModel):
    """
    Extracted information from the agent's answer about:
    - the novel claimed to have won both 2024 Pulitzer Prize for Fiction and 2024 National Book Award for Fiction
    - its publisher
    - URLs cited as evidence for each claim
    """
    title: Optional[str] = None
    author: Optional[str] = None

    # Award-specific source URLs
    pulitzer_sources: List[str] = Field(default_factory=list)  # URLs that support 2024 Pulitzer Prize for Fiction win
    nba_sources: List[str] = Field(default_factory=list)       # URLs that support 2024 National Book Award for Fiction win

    # Publisher and its supporting URLs
    publisher: Optional[str] = None
    publisher_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_book_and_publisher() -> str:
    return """
    Your goal is to extract the single primary novel that the answer claims won BOTH of the following in 2024:
    - the Pulitzer Prize for Fiction
    - the National Book Award for Fiction

    Extract the following fields from the answer text:
    1) title: the novel's title (string; if multiple novels are mentioned, choose the one the answer presents as winning both awards in 2024; otherwise choose the first clearly asserted novel).
    2) author: the author's name if present (string or null).
    3) pulitzer_sources: a list of all URLs explicitly cited in the answer that support that THIS novel won the 2024 Pulitzer Prize for Fiction. These may be official prize pages, credible news, Wikipedia, etc. Return only URLs that appear in the answer text.
    4) nba_sources: a list of all URLs explicitly cited in the answer that support that THIS novel won the 2024 National Book Award for Fiction. Return only URLs that appear in the answer text.
    5) publisher: the publisher name of the same novel, as stated in the answer (string or null). Prefer the specific imprint if given (e.g., "Alfred A. Knopf"); otherwise, the publishing house.
    6) publisher_sources: a list of all URLs explicitly cited in the answer that support the publisher attribution for THIS novel. Return only URLs that appear in the answer text.

    IMPORTANT URL RULES:
    - Extract only URLs that are explicitly present in the answer (including plain URLs or markdown links).
    - Do not invent or infer URLs.
    - If a URL is missing a protocol, prepend http://
    - If no URLs are provided for a category, return an empty list for that field.

    If any field is missing from the answer, set it to null (for strings) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip() != "")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_book_identification_subtree(
    evaluator: Evaluator,
    parent_node,
    info: BookAndPublisherInfo
) -> None:
    """
    Build and verify the subtree that checks:
    - a specific novel title is provided
    - the novel won the 2024 Pulitzer Prize for Fiction (supported by cited URLs)
    - the novel won the 2024 National Book Award for Fiction (supported by cited URLs)

    Structure:
      book_identification (parallel, non-critical)
        ├─ book_title_provided (custom leaf, critical)
        ├─ pulitzer_claim (parallel, non-critical)
        │    ├─ pulitzer_sources_provided (custom leaf, critical)
        │    └─ pulitzer_win_supported (leaf, critical)   [verify by URLs]
        └─ nba_claim (parallel, non-critical)
             ├─ nba_sources_provided (custom leaf, critical)
             └─ nba_win_supported (leaf, critical)        [verify by URLs]

    With root being sequential, the second major step (publisher) will be skipped unless
    this whole subtree aggregates to 1.0, i.e., both award claims pass and title provided.
    """
    node = evaluator.add_parallel(
        id="book_identification",
        desc="Correctly identify the novel that won both the Pulitzer Prize for Fiction and the National Book Award for Fiction in 2024",
        parent=parent_node,
        critical=False
    )

    # Title must be provided (critical gate)
    evaluator.add_custom_node(
        result=_nonempty(info.title),
        id="book_title_provided",
        desc="A novel title is provided in the answer",
        parent=node,
        critical=True
    )

    # Pulitzer claim subtree
    pulitzer_node = evaluator.add_parallel(
        id="pulitzer_claim",
        desc="The novel won the 2024 Pulitzer Prize for Fiction (supported by sources)",
        parent=node,
        critical=False
    )

    evaluator.add_custom_node(
        result=len(info.pulitzer_sources) > 0,
        id="pulitzer_sources_provided",
        desc="Pulitzer claim sources are provided",
        parent=pulitzer_node,
        critical=True
    )

    pulitzer_leaf = evaluator.add_leaf(
        id="pulitzer_win_supported",
        desc="The cited sources support that the novel won the 2024 Pulitzer Prize for Fiction",
        parent=pulitzer_node,
        critical=True
    )

    pulitzer_claim = f"The novel titled '{info.title or ''}' won the 2024 Pulitzer Prize for Fiction."
    await evaluator.verify(
        claim=pulitzer_claim,
        node=pulitzer_leaf,
        sources=info.pulitzer_sources if info.pulitzer_sources else None,
        additional_instruction="Verify that the provided webpage(s) explicitly indicate that this novel (exact or very close title match) is the 2024 winner of the Pulitzer Prize for Fiction. Allow minor formatting or punctuation differences in the title."
    )

    # National Book Award claim subtree
    nba_node = evaluator.add_parallel(
        id="nba_claim",
        desc="The novel won the 2024 National Book Award for Fiction (supported by sources)",
        parent=node,
        critical=False
    )

    evaluator.add_custom_node(
        result=len(info.nba_sources) > 0,
        id="nba_sources_provided",
        desc="National Book Award claim sources are provided",
        parent=nba_node,
        critical=True
    )

    nba_leaf = evaluator.add_leaf(
        id="nba_win_supported",
        desc="The cited sources support that the novel won the 2024 National Book Award for Fiction",
        parent=nba_node,
        critical=True
    )

    nba_claim = f"The novel titled '{info.title or ''}' won the 2024 National Book Award for Fiction."
    await evaluator.verify(
        claim=nba_claim,
        node=nba_leaf,
        sources=info.nba_sources if info.nba_sources else None,
        additional_instruction="Verify that the provided webpage(s) explicitly indicate that this novel (exact or very close title match) is the 2024 winner of the National Book Award for Fiction. Allow minor formatting or punctuation differences in the title."
    )


async def build_publisher_identification_subtree(
    evaluator: Evaluator,
    parent_node,
    info: BookAndPublisherInfo
) -> None:
    """
    Build and verify the subtree that checks:
    - publisher is provided
    - publisher is correctly supported by cited sources

    Structure:
      publisher_identification (parallel, non-critical)
        ├─ publisher_provided (custom leaf, critical)
        └─ publisher_claim (parallel, non-critical)
             ├─ publisher_sources_provided (custom leaf, critical)
             └─ publisher_supported_by_sources (leaf, critical)     [verify by URLs]
    """
    node = evaluator.add_parallel(
        id="publisher_identification",
        desc="Correctly identify the publisher of the identified novel",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_nonempty(info.publisher),
        id="publisher_provided",
        desc="Publisher is provided in the answer",
        parent=node,
        critical=True
    )

    pub_claim_node = evaluator.add_parallel(
        id="publisher_claim",
        desc="Publisher attribution is supported by cited sources",
        parent=node,
        critical=False
    )

    evaluator.add_custom_node(
        result=len(info.publisher_sources) > 0,
        id="publisher_sources_provided",
        desc="Publisher attribution sources are provided",
        parent=pub_claim_node,
        critical=True
    )

    pub_supported_leaf = evaluator.add_leaf(
        id="publisher_supported_by_sources",
        desc="The cited sources support the stated publisher for the identified novel",
        parent=pub_claim_node,
        critical=True
    )

    publisher_claim = f"The publisher of the novel titled '{info.title or ''}' is '{info.publisher or ''}'."
    await evaluator.verify(
        claim=publisher_claim,
        node=pub_supported_leaf,
        sources=info.publisher_sources if info.publisher_sources else None,
        additional_instruction=(
            "Verify that the webpage(s) explicitly indicate the book's publisher (or imprint). "
            "Treat imprints and their parent houses as acceptable equivalents if the page clearly indicates the relationship "
            "(e.g., 'Alfred A. Knopf' is an imprint of 'Knopf Doubleday Publishing Group'; both are acceptable labels depending on context). "
            "A page that clearly states 'Published by <publisher>' or 'Publisher: <publisher>' for the same book title should be considered supportive."
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
    Evaluate an answer for the task:
    Identify the publisher of the novel that won both the Pulitzer Prize for Fiction and the National Book Award for Fiction in 2024.
    """
    # Initialize evaluator (root is sequential: publisher step is skipped unless book step is perfect)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_book_and_publisher(),
        template_class=BookAndPublisherInfo,
        extraction_name="book_and_publisher_info"
    )

    # Build verification subtrees
    await build_book_identification_subtree(evaluator, root, extracted)
    await build_publisher_identification_subtree(evaluator, root, extracted)

    # Return the final structured evaluation summary
    return evaluator.get_summary()