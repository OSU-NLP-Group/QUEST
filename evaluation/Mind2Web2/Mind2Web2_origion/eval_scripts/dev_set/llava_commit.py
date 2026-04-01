import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.llm_client.base_client import LLMClient
from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy

TASK_ID = "llava_commit"
TASK_DESCRIPTION = """
Identify the first commit on the main branch of the official Hugging Face transformers repository that added support for the LLaVA model. Please provide the following details about this commit: the short commit ID (first 7 characters), the date of the commit, a list of all contributors/authors involved in this commit. For each author, include a link to their GitHub profile page and the full real name displayed on their GitHub profile page.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {
    "commit_id": "44b5506",
    "date": "Dec 7, 2023",
    "expected_authors": [
        "Younes B",
        "Arthur",
        "Shauray Singh",
        "Lysandre Debut",
        "Haotian Liu"
    ]
}


class AuthorInfo(BaseModel):
    """Data model for individual author information."""
    name: Optional[str] = Field(default=None, description="Author name as provided in the answer")
    profile_url: Optional[str] = Field(default=None, description="GitHub profile page URL")
    real_name_from_profile: Optional[str] = Field(default=None, description="Real name extracted from profile page")


class CommitInfo(BaseModel):
    """Data model for extracted commit information."""
    commit_id: Optional[str] = Field(default=None, description="Short commit ID (7 characters)")
    date: Optional[str] = Field(default=None, description="Date of the commit")
    source_urls: Optional[List[str]] = Field(default_factory=list, description="Source URLs for commit verification")


class AuthorsInfo(BaseModel):
    """Data model for extracted authors information."""
    authors: Optional[List[AuthorInfo]] = Field(default_factory=list,
                                                description="List of authors with their profile info")


def prompt_extract_commit_info() -> str:
    """
    Extraction prompt for getting basic commit information from the answer.
    """
    return """
    Extract the basic commit information for the LLaVA model support from the answer.

    Look for:
    - commit_id: The short commit ID (typically 7 characters) 
    - date: The date when the commit was made
    - source_urls: Any URLs that contain or reference this commit information (e.g., GitHub commit URLs, repository links)

    Extract the information exactly as it appears in the text.
    If any field is not mentioned, set it to null or empty list.
    """


def prompt_extract_authors_info() -> str:
    """
    Extraction prompt for getting authors information from the answer.
    """
    return """
    Extract all author information mentioned in the answer related to the LLaVA commit.

    For each author, extract:
    - name: The author's name as mentioned in the answer
    - profile_url: Their GitHub profile page URL if provided
    - real_name_from_profile: Their real name as stated to appear on their profile page

    Extract information exactly as it appears in the text.
    If any field is not mentioned for an author, set it to null.
    Include all authors mentioned, even if some information is incomplete.
    """


async def verify_commit_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        commit_info: CommitInfo,
) -> None:
    """
    Verify commit information in sequential order.
    This is the first node in the sequential chain.
    """
    # Create sequential node for commit verification steps
    commit_verification = evaluator.add_sequential(
        id="commit_verification",
        desc="Verify commit ID, date, and provenance in sequence",
        parent=parent_node,
        critical=False  # Non-critical to allow sequential partial scoring
    )

    # Step 1: Verify commit ID exists and is correct
    await verify_commit_id(evaluator, commit_verification, commit_info)

    # Step 2: Verify commit date
    await verify_commit_date(evaluator, commit_verification, commit_info)


async def verify_commit_id(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        commit_info: CommitInfo,
) -> None:
    """
    Verify commit ID existence, correctness and provenance.
    """
    # Create parallel node for commit ID checks
    commit_id_node = evaluator.add_parallel(
        id="commit_id_verification",
        desc="Verify commit ID existence, correctness and provenance",
        parent=parent_node,
        critical=True
    )

    # Check if commit ID exists
    id_exists = evaluator.add_custom_node(
        result=bool(commit_info.commit_id and commit_info.commit_id.strip() and commit_info.source_urls),
        id="commit_id_exists",
        desc="Commit ID and source URLs are provided in the answer",
        parent=commit_id_node,
        critical=True,
    )

    # Verify commit ID correctness
    id_correctness = evaluator.add_leaf(
        id="commit_id_correctness",
        desc="Commit ID matches the expected value (44b5506)",
        parent=commit_id_node,
        critical=True,  # Critical - ID must be correct
    )

    # Always perform verification - short-circuit logic will skip if existence failed
    claim = f"This ID (a github commit id) '{commit_info.commit_id or 'N/A'}' matches this ID '{GROUND_TRUTH['commit_id']}'"
    await evaluator.verify(
        claim=claim,
        node=id_correctness,
        sources=None,  # Simple comparison, no URL needed
        additional_instruction="Allow minor formatting differences or extra descriptions but the core 7-character commit ID should exist and match exactly. Expected: 44b5506."
    )

    # Provenance check - verify the commit info is supported by sources
    provenance_check = evaluator.add_leaf(
        id="commit_provenance",
        desc="Commit information is supported by provided source URLs",
        parent=commit_id_node,
        critical=True,  # Critical - information must be substantiated
    )

    # Always perform verification - short-circuit logic will skip if needed
    claim = f"This page shows or mentioned the github commit ID: '{commit_info.commit_id}'. For example, if this is exactly the commit page"
    await evaluator.verify(
        claim=claim,
        node=provenance_check,
        sources=commit_info.source_urls,
    )


async def verify_commit_date(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        commit_info: CommitInfo,
) -> None:
    """
    Verify commit date accuracy.
    """

    commit_date_node = evaluator.add_parallel(
        id="commit_date_verification",
        desc="Verify commit date existence, correctness and provenance",
        parent=parent_node,
        critical=True
    )

    # Check if date exists
    date_exists = evaluator.add_custom_node(
        result=bool(commit_info.date and commit_info.date.strip()),
        id="commit_date_exists",
        desc="Commit date is provided in the answer",
        parent=commit_date_node,
        critical=True,  # Critical - date is required
    )

    # Verify date correctness
    date_correctness = evaluator.add_leaf(
        id="commit_date_correctness",
        desc="Commit date matches the expected value (Dec 7, 2023)",
        parent=commit_date_node,
        critical=True,  # Critical - date must be correct
    )

    # Always perform verification - short-circuit logic will skip if existence failed
    claim = f"The provided commit date '{commit_info.date or 'N/A'}' matches the expected date '{GROUND_TRUTH['date']}'"
    await evaluator.verify(
        claim=claim,
        node=date_correctness,
        sources=None,
        additional_instruction="Allow reasonable date format variations (e.g., 'Dec 7, 2023', 'December 7, 2023', '2023-12-07') but the core date should match. Expected: Dec 7, 2023."
    )

    # Provenance check - verify the commit date is supported by sources
    provenance_check = evaluator.add_leaf(
        id="commit_date_provenance",
        desc="Commit date information is supported by provided source URLs",
        parent=commit_date_node,
        critical=True,  # Critical - information must be substantiated
    )

    # Always perform verification - short-circuit logic will skip if needed
    claim = f"This page shows or mentioned the date of the github commit ID '{commit_info.commit_id}' is {commit_info.date}."
    await evaluator.verify(
        claim=claim,
        node=provenance_check,
        sources=commit_info.source_urls,
        additional_instruction="Allow reasonable date format variations (e.g., 'Dec 7, 2023', 'December 7, 2023', '2023-12-07') but the core date should match."
    )


async def verify_authors_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        authors_info: AuthorsInfo,
        commit_info: CommitInfo,
) -> None:
    """
    Verify authors information in parallel.
    This is the second node in the sequential chain.
    """
    # Create parallel node for authors verification
    authors_verification = evaluator.add_parallel(
        id="authors_verification",
        desc="Verify all authors information in parallel",
        parent=parent_node,
        critical=False  # Non-critical to allow partial scoring
    )

    # Extract first 5 authors, pad with empty authors if needed
    provided_authors = authors_info.authors[:5] if authors_info.authors else []

    # Pad with empty AuthorInfo objects for missing authors
    while len(provided_authors) < 5:
        provided_authors.append(AuthorInfo())  # Empty author info

    # Create verification nodes for each author position
    for i, author in enumerate(provided_authors):
        await verify_single_author(evaluator, authors_verification, author, i + 1, commit_info)


async def verify_single_author(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        author: AuthorInfo,
        author_number: int,
        commit_info: CommitInfo,
) -> None:
    """
    Verify a single author's information.
    """
    # Create parallel node for this author
    author_node = evaluator.add_parallel(
        id=f"author_{author_number}",
        desc=f"Author {author_number} information verification",
        parent=parent_node,
        critical=False  # Non-critical to allow partial scoring across authors
    )

    # Check if author information exists
    author_exists = evaluator.add_custom_node(
        result=bool(author.name and author.name.strip() and author.profile_url),
        id=f"author_{author_number}_exists",
        desc=f"Author {author_number} name and profile URL are provided",
        parent=author_node,
        critical=True
    )

    # Verify name matches expected contributors
    name_match_node = evaluator.add_leaf(
        id=f"author_{author_number}_name_match",
        desc=f"Author {author_number} name matches one of the expected contributors",
        parent=author_node,
        critical=True,  # Critical - must match an expected author
    )

    # Always perform verification - short-circuit logic will skip if existence failed
    expected_authors_str = ", ".join(GROUND_TRUTH['expected_authors'])
    author_name = author.real_name_from_profile if author.real_name_from_profile else author.name
    claim = f"The name '{author_name or 'N/A'}' matches one of the names in the following list: {expected_authors_str}"
    await evaluator.verify(
        claim=claim,
        node=name_match_node,
        sources=None,  # Simple name matching
        additional_instruction="Perform case insensitive matching. Allow variations like 'Arthur' matching 'Arthur Zucker', or reasonable name format differences. Expected authors: Younes B, Arthur (or Arthur Zucker), Shauray Singh, Lysandre Debut, Haotian Liu."
    )

    # Verify profile page is provided (non-critical)
    profile_provided_node = evaluator.add_leaf(
        id=f"author_{author_number}_profile_provided",
        desc=f"Author {author_number} GitHub profile page URL is provided",
        parent=author_node,
        critical=True,
    )

    profile_claim = f"This is a GitHub profile page for '{author.name or 'N/A'}'"
    await evaluator.verify(
        claim=profile_claim,
        node=profile_provided_node,
        sources=author.profile_url,
    )

    # Provenance check 
    author_provenance_node = evaluator.add_leaf(
        id=f"author_{author_number}_commit_provenance",
        desc=f"Author {author_number} is shown as a contributor to the commit",
        parent=author_node,
        critical=True,  # Critical - must be actually associated with the commit
    )

    provenance_claim = f"The page shows that '{author.name or 'N/A'}' is a contributor, co-author, or author of the commit {commit_info.commit_id}"
    await evaluator.verify(
        claim=provenance_claim,
        node=author_provenance_node,
        sources=commit_info.source_urls,
        additional_instruction="Verify that this specific person is listed as a contributor, co-author, or commit author on the commit page. Look for author attributions, 'Co-authored-by' tags, or contributor listings that include this person's name."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                               #
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
    Main evaluation function for the LLaVA commit finding task.

    This function implements a sequential verification strategy:
    1. First verify commit information (ID, date, provenance)
    2. Then verify authors information (only if commit info is correct)

    The sequential design ensures that if commit information is wrong,
    author verification is automatically skipped via short-circuit logic.
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential: commit info → authors info
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Record ground truth information
    evaluator.add_ground_truth(GROUND_TRUTH, "expected_commit_and_authors_info")

    # -------- 2. Extract structured information ------------------- #

    # Extract basic commit information
    commit_info = await evaluator.extract(
        prompt=prompt_extract_commit_info(),
        template_class=CommitInfo,
        extraction_name="commit_extraction",
        source=None,  # Extract from answer text
    )

    # Extract authors information
    authors_info = await evaluator.extract(
        prompt=prompt_extract_authors_info(),
        template_class=AuthorsInfo,
        extraction_name="authors_extraction",
        source=None,  # Extract from answer text
    )

    # -------- 3. Build verification tree (Sequential) ------------- #

    # Step 1: Verify commit information (non-critical for sequential scoring)
    await verify_commit_info(evaluator, root, commit_info)

    # Step 2: Verify authors information (will be skipped if commit info fails)
    await verify_authors_info(evaluator, root, authors_info, commit_info)

    # -------- 4. Return evaluation results ------------------------ #
    return evaluator.get_summary()