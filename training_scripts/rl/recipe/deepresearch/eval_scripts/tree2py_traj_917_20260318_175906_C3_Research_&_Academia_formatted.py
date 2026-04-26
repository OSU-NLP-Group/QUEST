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
TASK_ID = "spinosaurus_prep"
TASK_DESCRIPTION = (
    "A graduate student in paleontology is preparing to submit a research paper about a recently discovered dinosaur "
    "species to a high-impact journal. They want to use the February 2026 publication about Spinosaurus mirabilis as a "
    "model for their submission. To properly prepare their manuscript, they need to: (1) Identify which peer-reviewed "
    "journal published the Spinosaurus mirabilis research paper (with the distinctive scimitar-shaped crest) in "
    "February 2026, (2) Provide the DOI for this publication, (3) Identify the lead (first) author of this paper and "
    "verify their institutional affiliation, (4) Determine the maximum word count allowed for abstracts in Research "
    "Articles submitted to this journal, and (5) Provide a reference URL that documents the abstract word count "
    "requirement. What information should the graduate student provide to complete their preparation?"
)

EXPECTED = {
    "journal_name": "Science (AAAS)",  # Accept 'Science' / 'Science journal' / 'Science Magazine (AAAS)'
    "doi": "10.1126/science.adx5486",
    "lead_author": "Paul C. Sereno",
    "affiliation": "University of Chicago, Department of Organismal Biology and Anatomy",
    "abstract_word_limit": "125 words or fewer"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PrepExtraction(BaseModel):
    # Publication details mentioned in the answer
    journal_name: Optional[str] = None
    doi: Optional[str] = None
    lead_author: Optional[str] = None
    affiliation: Optional[str] = None
    publication_urls: List[str] = Field(default_factory=list)

    # Submission requirements (abstract length) and its documentation URLs
    abstract_word_limit: Optional[str] = None
    requirement_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_prep_info() -> str:
    return """
    Extract the specific items the answer provides for preparing the manuscript using the February 2026
    Spinosaurus mirabilis paper as a model. Only extract information explicitly stated in the answer text.
    Do not invent anything. If an item is missing, set it to null (or [] for lists).

    Required fields to extract:
    1) journal_name: The name of the journal that published the Spinosaurus mirabilis paper.
       Examples of acceptable forms include "Science", "Science (AAAS)", or similar synonymous naming used in the answer.
    2) doi: The DOI string for the publication (e.g., "10.1126/science.adx5486"). Extract exactly as written.
    3) lead_author: The first/lead author's full name as given in the answer.
    4) affiliation: The institutional affiliation of the lead (first) author as stated in the answer.
    5) publication_urls: A list of all URLs in the answer that specifically refer to the Spinosaurus paper
       (e.g., the Science article page, publisher page, Google Scholar, DOI resolver link).
       Only include actual URLs that appear in the answer.
    6) abstract_word_limit: The abstract word-count limit for Research Articles in the referenced journal,
       exactly as the answer states it (e.g., "125 words", "≤125 words", "no more than 125 words").
    7) requirement_urls: A list of URLs in the answer that document the abstract word-count requirement.
       These should be official policy or author guideline pages when possible. Only include actual URLs
       present in the answer text.

    Return a single JSON object with keys:
    {
      "journal_name": string|null,
      "doi": string|null,
      "lead_author": string|null,
      "affiliation": string|null,
      "publication_urls": string[],
      "abstract_word_limit": string|null,
      "requirement_urls": string[]
    }
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_paper_identification(
    evaluator: Evaluator,
    parent_node,
    info: PrepExtraction
) -> None:
    """
    Build the 'Paper_Identification' subtree and run all verification leaves.
    """
    paper_ident_node = evaluator.add_parallel(
        id="paper_identification",
        desc="Identify all publication details for the Spinosaurus mirabilis paper",
        parent=parent_node,
        critical=False
    )

    # Journal details group
    journal_group = evaluator.add_parallel(
        id="journal_details",
        desc="Identify the journal and DOI for the publication",
        parent=paper_ident_node,
        critical=False
    )

    # Journal name leaf (critical)
    journal_leaf = evaluator.add_leaf(
        id="journal_name",
        desc="Identify that the paper was published in Science journal (AAAS)",
        parent=journal_group,
        critical=True
    )
    journal_claim = (
        "The Spinosaurus mirabilis research paper referenced in the answer was published in the journal "
        "Science (AAAS). Treat 'Science', 'Science journal', or 'Science Magazine (AAAS)' as equivalent names."
    )
    await evaluator.verify(
        claim=journal_claim,
        node=journal_leaf,
        sources=info.publication_urls,
        additional_instruction=(
            "Use the provided publication URLs (e.g., Science article page, DOI landing page) to confirm "
            "the journal is 'Science'. Accept reasonable synonyms such as 'Science', 'Science (AAAS)', or "
            "'Science Magazine by AAAS'. Do NOT accept 'Science Advances' or other Science family journals."
        )
    )

    # DOI leaf (critical)
    doi_leaf = evaluator.add_leaf(
        id="publication_doi",
        desc="Provide the DOI 10.1126/science.adx5486",
        parent=journal_group,
        critical=True
    )
    doi_claim = (
        "The DOI of the Spinosaurus mirabilis Science paper is 10.1126/science.adx5486 "
        "(case-insensitive comparison for the DOI string is acceptable)."
    )
    await evaluator.verify(
        claim=doi_claim,
        node=doi_leaf,
        sources=info.publication_urls,
        additional_instruction=(
            "Verify the DOI string on the provided publication URL(s). If the page shows a DOI, check it matches "
            "'10.1126/science.adx5486' allowing case-insensitive match. If the provided URL is a DOI resolver page, "
            "ensure it resolves for the same DOI."
        )
    )

    # Author details group
    author_group = evaluator.add_parallel(
        id="author_details",
        desc="Identify lead author and institutional affiliation",
        parent=paper_ident_node,
        critical=False
    )

    # Lead author name (critical)
    lead_author_leaf = evaluator.add_leaf(
        id="lead_author_name",
        desc="Identify Paul C. Sereno as the lead (first) author",
        parent=author_group,
        critical=True
    )
    lead_author_claim = "The first (lead) author of the Spinosaurus mirabilis Science paper is Paul C. Sereno."
    await evaluator.verify(
        claim=lead_author_claim,
        node=lead_author_leaf,
        sources=info.publication_urls,
        additional_instruction=(
            "Confirm the first-listed author on the article page is 'Paul C. Sereno'. Minor formatting or middle "
            "initial variations are acceptable (e.g., 'Paul Sereno' vs 'Paul C. Sereno')."
        )
    )

    # Institutional affiliation (critical)
    affiliation_leaf = evaluator.add_leaf(
        id="institutional_affiliation",
        desc="Verify affiliation as University of Chicago, Department of Organismal Biology and Anatomy",
        parent=author_group,
        critical=True
    )
    affiliation_claim = (
        "On the paper, Paul C. Sereno's institutional affiliation is University of Chicago, Department of "
        "Organismal Biology and Anatomy (or 'Organismal Biology & Anatomy')."
    )
    await evaluator.verify(
        claim=affiliation_claim,
        node=affiliation_leaf,
        sources=info.publication_urls,
        additional_instruction=(
            "Check the author affiliations listed on the article page. Allow small formatting variations such as '&' "
            "vs 'and', inclusion or omission of 'Department of', or 'The University of Chicago'."
        )
    )


async def build_and_verify_submission_requirements(
    evaluator: Evaluator,
    parent_node,
    info: PrepExtraction
) -> None:
    """
    Build the 'Journal_Submission_Requirements' subtree and run all verification leaves.
    """
    req_node = evaluator.add_parallel(
        id="journal_submission_requirements",
        desc="Determine submission requirements for Science journal",
        parent=parent_node,
        critical=False
    )

    # Abstract word count (critical)
    abstract_wc_leaf = evaluator.add_leaf(
        id="abstract_word_count",
        desc="Identify that Science requires abstracts to be 125 words or less for Research Articles",
        parent=req_node,
        critical=True
    )
    abstract_wc_claim = (
        "Science journal's author guidelines for Research Articles state that abstracts must be 125 words or fewer."
    )
    await evaluator.verify(
        claim=abstract_wc_claim,
        node=abstract_wc_leaf,
        sources=info.requirement_urls,
        additional_instruction=(
            "Focus on policy pages for Science (the flagship journal) author guidelines or submission instructions. "
            "Accept phrasing such as 'no more than 125 words', '≤125 words', or '125 words maximum' for Research "
            "Articles' abstracts. Do not confuse with Science Advances or other journals in the Science family."
        )
    )

    # Documentation URL (critical)
    doc_url_leaf = evaluator.add_leaf(
        id="documentation_url",
        desc="Provide valid reference URL documenting the abstract word count requirement",
        parent=req_node,
        critical=True
    )
    doc_url_claim = (
        "This URL is an official Science (science.org) author instructions/guidelines page that documents the abstract "
        "length requirement for Research Articles."
    )
    await evaluator.verify(
        claim=doc_url_claim,
        node=doc_url_leaf,
        sources=info.requirement_urls,
        additional_instruction=(
            "Verify that at least one provided URL is an official Science/AAAS page (science.org domain preferred) "
            "that explicitly documents author instructions or policies including abstract length for Research Articles."
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
    Evaluate an answer for the Spinosaurus mirabilis manuscript preparation task.
    """
    # Initialize evaluator (root is non-critical by design of framework)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Follow the rubric's sequential top-level aggregation
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

    # Extract structured fields from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_prep_info(),
        template_class=PrepExtraction,
        extraction_name="prep_extraction"
    )

    # Record ground-truth expectations for transparency (not used for scoring directly)
    evaluator.add_ground_truth({
        "expected_journal": EXPECTED["journal_name"],
        "expected_doi": EXPECTED["doi"],
        "expected_lead_author": EXPECTED["lead_author"],
        "expected_affiliation": EXPECTED["affiliation"],
        "expected_abstract_limit": EXPECTED["abstract_word_limit"]
    })

    # Build and verify tree according to rubric
    # Root node acts as "Manuscript_Preparation" (sequential)
    # Child 1: Paper Identification (parallel)
    await build_and_verify_paper_identification(evaluator, root, extracted_info)

    # Child 2: Journal Submission Requirements (parallel)
    await build_and_verify_submission_requirements(evaluator, root, extracted_info)

    # Return final structured evaluation summary
    return evaluator.get_summary()