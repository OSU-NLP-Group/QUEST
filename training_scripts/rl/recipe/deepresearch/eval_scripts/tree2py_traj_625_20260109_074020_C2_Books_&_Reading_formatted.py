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
TASK_ID = "debut_award_2020_indie_publisher"
TASK_DESCRIPTION = (
    "Identify the title and author of a debut novel that won a major literary award (either the Booker Prize or the "
    "National Book Award for Fiction) in 2020, where the author was born in the 1970s, and the novel was published by "
    "an independent publisher that was founded (or whose founding constituent companies were established) before 1980. "
    "Provide the novel's title, the author's name, the specific award won, the publisher's name, and the publisher's "
    "founding year."
)

ALLOWED_AWARDS = [
    "Booker Prize",
    "National Book Award for Fiction",
    "National Book Award (Fiction)",
    "NBA for Fiction",
    "The Booker Prize",
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NovelSourceInfo(BaseModel):
    award_urls: List[str] = Field(default_factory=list)
    debut_urls: List[str] = Field(default_factory=list)
    author_birth_urls: List[str] = Field(default_factory=list)
    publisher_urls: List[str] = Field(default_factory=list)
    general_urls: List[str] = Field(default_factory=list)


class NovelExtraction(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    award: Optional[str] = None
    award_year: Optional[str] = None
    publisher: Optional[str] = None
    publisher_founding_year: Optional[str] = None
    sources: NovelSourceInfo = Field(default_factory=NovelSourceInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_novel_info() -> str:
    return (
        "Extract the following from the answer exactly as stated:\n"
        "1. title: The novel's title.\n"
        "2. author: The author's full name.\n"
        "3. award: The specific award the novel won (e.g., 'Booker Prize' or 'National Book Award for Fiction').\n"
        "4. award_year: The year of the award (e.g., '2020'). If not explicitly stated, return null.\n"
        "5. publisher: The independent publisher's name that published the novel.\n"
        "6. publisher_founding_year: The publisher's founding year (or the earliest founding year among its founding constituent companies), as provided in the answer. If absent, return null.\n"
        "\n"
        "Additionally, extract URLs cited in the answer that support each claim:\n"
        "- sources.award_urls: URLs supporting the award and year for the novel.\n"
        "- sources.debut_urls: URLs supporting that the novel is the author's debut novel.\n"
        "- sources.author_birth_urls: URLs supporting that the author was born in the 1970s.\n"
        "- sources.publisher_urls: URLs supporting publisher independence and founding year before 1980.\n"
        "- sources.general_urls: Any other URLs cited that relate to the novel or author.\n"
        "\n"
        "Rules:\n"
        "- Return only information explicitly present in the answer; do not infer.\n"
        "- For URLs, include full valid URLs mentioned (plain or markdown). If absent, use an empty list.\n"
        "- If any field is missing, set it to null.\n"
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_str(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    extracted: NovelExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # Create the critical parallel node for the main task
    novel_node = evaluator.add_parallel(
        id="Novel_Identification",
        desc="The response identifies a novel and provides required attributes that satisfy all specified criteria.",
        parent=root_node,
        critical=True,
    )

    # Existence / completeness check to gate subsequent verifications
    required_present = (
        bool(_safe_str(extracted.title)) and
        bool(_safe_str(extracted.author)) and
        bool(_safe_str(extracted.award)) and
        bool(_safe_str(extracted.publisher)) and
        bool(_safe_str(extracted.publisher_founding_year))
    )

    evaluator.add_custom_node(
        result=required_present,
        id="Required_Attributes_Present",
        desc="Required fields (title, author, award, publisher, publisher founding year) are provided in the answer.",
        parent=novel_node,
        critical=True
    )

    # Award verification (single leaf, critical)
    award_leaf = evaluator.add_leaf(
        id="Award_Won_In_2020",
        desc="The novel won either the Booker Prize or the National Book Award for Fiction in 2020.",
        parent=novel_node,
        critical=True,
    )

    title = _safe_str(extracted.title)
    author = _safe_str(extracted.author)
    award = _safe_str(extracted.award)
    award_year = _safe_str(extracted.award_year)

    award_claim = (
        f"The novel '{title}' by {author} won either the Booker Prize or the National Book Award for Fiction in 2020. "
        f"The answer indicates the specific award as '{award}'"
        f"{' and year ' + award_year if award_year else ''}."
    )

    await evaluator.verify(
        claim=award_claim,
        node=award_leaf,
        sources=extracted.sources.award_urls or extracted.sources.general_urls,
        additional_instruction=(
            "Verify using the provided URLs that this specific novel won either the Booker Prize or the National Book "
            "Award for Fiction in 2020. Accept reasonable naming variants (e.g., 'National Book Award (Fiction)' or "
            "'The Booker Prize'). If sources show a different year or a different award category, mark as not supported."
        ),
    )

    # Debut novel status verification (single leaf, critical)
    debut_leaf = evaluator.add_leaf(
        id="Debut_Novel_Status",
        desc="The novel is the author's debut novel (first published novel).",
        parent=novel_node,
        critical=True,
    )

    debut_claim = f"The novel '{title}' is the debut (first published novel) of {author}."
    await evaluator.verify(
        claim=debut_claim,
        node=debut_leaf,
        sources=extracted.sources.debut_urls or extracted.sources.general_urls,
        additional_instruction=(
            "Confirm that this novel is the author's first published novel. If the author previously published another "
            "novel before this one, this claim is not supported. Short story collections or non-fiction do not count "
            "as a prior 'novel'."
        ),
    )

    # Author birth year in the 1970s (single leaf, critical)
    birth_leaf = evaluator.add_leaf(
        id="Author_Birth_Year",
        desc="The author was born in the 1970s (1970–1979 inclusive).",
        parent=novel_node,
        critical=True,
    )

    birth_claim = (
        f"The author {author} was born in the 1970s (1970–1979 inclusive). "
        f"{'The answer mentions ' + extracted.publisher_founding_year if False else ''}"
    )

    await evaluator.verify(
        claim=birth_claim,
        node=birth_leaf,
        sources=extracted.sources.author_birth_urls or extracted.sources.general_urls,
        additional_instruction=(
            "Check the author’s birth year on the provided source(s). Consider the claim supported only if the year "
            "falls between 1970 and 1979 inclusive."
        ),
    )

    # Publisher requirements (parallel, critical) with two critical children
    pub_node = evaluator.add_parallel(
        id="Publisher_Requirements",
        desc="The novel was published by a qualifying independent publisher founded (or with constituent companies established) before 1980.",
        parent=novel_node,
        critical=True,
    )

    publisher = _safe_str(extracted.publisher)
    founding_year = _safe_str(extracted.publisher_founding_year)

    # Independent publisher check
    indie_leaf = evaluator.add_leaf(
        id="Independent_Publisher",
        desc="The publisher is independent (not one of the Big Five: Penguin Random House, HarperCollins, Simon & Schuster, Hachette, Macmillan).",
        parent=pub_node,
        critical=True,
    )

    indie_claim = (
        f"The publisher '{publisher}' is an independent publisher and is not one of the Big Five (Penguin Random House, "
        f"HarperCollins, Simon & Schuster, Hachette, Macmillan)."
    )

    await evaluator.verify(
        claim=indie_claim,
        node=indie_leaf,
        sources=extracted.sources.publisher_urls or extracted.sources.general_urls,
        additional_instruction=(
            "Verify that the publisher is independent and not part of the Big Five. If the source indicates that it is "
            "an imprint or division of any Big Five company, mark as not supported."
        ),
    )

    # Publisher founded before 1980 check
    founded_leaf = evaluator.add_leaf(
        id="Publisher_Founded_Before_1980",
        desc="The independent publisher or its founding constituent companies were established before 1980.",
        parent=pub_node,
        critical=True,
    )

    founded_claim = (
        f"The publisher '{publisher}' was founded before 1980, specifically in {founding_year}. If the current entity "
        f"was formed later, its founding constituent companies were established before 1980."
    )

    await evaluator.verify(
        claim=founded_claim,
        node=founded_leaf,
        sources=extracted.sources.publisher_urls or extracted.sources.general_urls,
        additional_instruction=(
            "Verify that the founding year is before 1980. If the publisher's current corporate entity was formed after "
            "1980, accept the claim if the founding constituent company (e.g., an earlier press merged into the current "
            "company) was established before 1980 and the provided sources clearly indicate this."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the debut novel award 2020 with independent publisher requirements.
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

    # Extract structured novel info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_novel_info(),
        template_class=NovelExtraction,
        extraction_name="novel_info",
    )

    # Optional: add ground truth context list for allowed awards (for transparency in report)
    evaluator.add_ground_truth({
        "allowed_awards_examples": ALLOWED_AWARDS,
        "award_year_requirement": "2020",
        "big_five": [
            "Penguin Random House",
            "HarperCollins",
            "Simon & Schuster",
            "Hachette",
            "Macmillan"
        ],
        "publisher_year_requirement": "Founded (or founding constituent companies) before 1980"
    }, gt_type="constraints_reference")

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()