import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "rag_repo"
TASK_DESCRIPTION = """
I am interested in Retrieval-Augmented Generation (RAG) and would like to explore popular and easy-to-use repositories on GitHub. Please identify the 10 most-starred repositories listed under the GitHub topic "retrieval-augmented-generation". For each repository, please provide its name, a direct link to its GitHub page, and indicate whether its README explicitly states that it can be installed using a single pip install <package-name> command (without additional steps). If it does, please also provide a link to its PyPI page; if not, explicitly note that this installation method isn't mentioned.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}

# Ground truth URL for the top RAG repositories
GROUND_TRUTH_URL = "https://github.com/topics/retrieval-augmented-generation"


class RepositoryInfo(BaseModel):
    """Information about a single repository"""
    name: Optional[str] = Field(default=None,
                                description="Repository name as mentioned in the answer (e.g., 'xxxrag from THUNLP')")
    github_url: Optional[str] = Field(default=None, description="GitHub repository URL")
    pip_install_explanation: Optional[str] = Field(default=None,
                                                   description="Any explanation about pip installation from the answer")
    pypi_url: Optional[str] = Field(default=None, description="PyPI page URL if provided")
    can_pip_install: bool = Field(default=False,
                                  description="Whether the answer states it can be pip installed (True) or not (False)")


class ExtractedRepositories(BaseModel):
    """All extracted repository information"""
    repositories: List[RepositoryInfo] = Field(default_factory=list,
                                               description="List of repositories in order they appear")


class TopRepositoriesList(BaseModel):
    """Ground truth top repositories from GitHub topic page"""
    top_10_repo_full_names: List[str] = Field(default_factory=list, description="Full names including org/repo format")


def prompt_extract_repositories() -> str:
    """Extraction prompt for repository information"""
    return """
    Extract information about RAG (Retrieval-Augmented Generation) repositories from the answer.

    For each repository mentioned, extract IN THE ORDER THEY APPEAR:
    - name: The repository name EXACTLY as mentioned in the answer (e.g., if it says "xxxrag from THUNLP", extract exactly that)
    - github_url: The GitHub URL for the repository
    - pip_install_explanation: Any explanation or note about pip installation from the answer
    - pypi_url: The PyPI page URL if provided
    - can_pip_install: 
      * Set to True ONLY if the answer explicitly states it can be installed with a single "pip install <package-name>" command
      * Set to False for all other cases including:
        - When it says pip install is not available
        - When it mentions "pip install -e ." or other non-PyPI installation methods
        - When no pip installation information is provided
        - When installation requires additional steps beyond a single pip command

    Extract all repositories mentioned, preserving their exact order in the answer.
    Important: can_pip_install must be either True or False, never null.
    """


def prompt_extract_top_repos() -> str:
    """Extraction prompt for ground truth top repositories"""
    return """
    You are looking at the GitHub topic page for "retrieval-augmented-generation" which shows repositories sorted by stars.

    Extract the FULL names of the TOP 10 most-starred repositories shown on this page.

    IMPORTANT: Extract the complete repository name including the organization/user prefix.
    For example: "langchain-ai/langchain" or "infiniflow/ragflow" (not just "langchain" or "ragflow")

    The repositories should be listed in order from most to least stars.
    Extract exactly 10 repository full names in "org/repo" format.
    """


async def verify_repository(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        repo_info: RepositoryInfo,
        repo_index: int,
        ground_truth_repos_string: str,
) -> None:
    """Verify a single repository's information"""

    # Create a sequential node for this repository (non-critical for partial scoring)
    repo_node = evaluator.add_sequential(
        id=f"repo_{repo_index}",
        desc=f"Repository {repo_index + 1}: {repo_info.name or 'Empty'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial scoring
    )

    # 1. Repository information verification (non-critical)
    repo_info_node = evaluator.add_parallel(
        id=f"repo_{repo_index}_info",
        desc=f"Repository information for {repo_info.name or 'Empty'}",
        parent=repo_node,
        critical=False,
    )

    # Check if all repository info exists
    info_exists = evaluator.add_custom_node(
        result=bool(
            repo_info.name and
            repo_info.github_url and
            repo_info.can_pip_install is not None  # This is always True/False, so just check it's not None
        ),
        id=f"repo_{repo_index}_info_exists",
        desc="Repository has all required information (name, GitHub URL, pip install status)",
        parent=repo_info_node,
        critical=True,
    )

    # Verify repository is in ground truth list
    # if repo_info.name:
    in_gt_node = evaluator.add_leaf(
        id=f"repo_{repo_index}_in_ground_truth",
        desc=f"Repository '{repo_info.name}' is in the top 10 most-starred RAG repositories",
        parent=repo_info_node,
        critical=True,
    )

    claim = f"The repository '{repo_info.name}' appears in this repo list: {ground_truth_repos_string}"
    await evaluator.verify(
        claim=claim,
        node=in_gt_node,
        additional_instruction="Check if the repository name matches any in the list. Allow for variations like different organization names, hyphens vs underscores, or partial name matches, or formatting. As long as it essentially matches one of the repos in the list, it is considered correct."
    )

    # Verify GitHub URL corresponds to the repository
    # if repo_info.github_url and repo_info.name:
    github_url_node = evaluator.add_leaf(
        id=f"repo_{repo_index}_github_url_matches",
        desc=f"GitHub URL corresponds to repository '{repo_info.name}'",
        parent=repo_info_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"This is a GitHub page and this is for the repository '{repo_info.name}'",
        node=github_url_node,
        sources=repo_info.github_url,
        additional_instruction="Verify the URL leads to a repository that matches the given name. Allow for organization prefixes and naming variations."
    )

    # 2. Pip install information verification (sequential, non-critical)
    pip_info_node = evaluator.add_sequential(
        id=f"repo_{repo_index}_pip_info",
        desc=f"Pip installation information for {repo_info.name or 'Empty'}",
        parent=repo_node,
        critical=False,
    )

    # Verify pip install claim against README
    # if repo_info.github_url:
    pip_claim_node = evaluator.add_leaf(
        id=f"repo_{repo_index}_pip_claim_accurate",
        desc=f"Pip installation claim matches README content",
        parent=pip_info_node,
        critical=True,
    )

    # Construct README URL
    readme_url = repo_info.github_url.rstrip('/') + "/blob/main/README.md" if repo_info.github_url else []

    if repo_info.can_pip_install:
        claim = f"The README shows that this repository can be installed with a single 'pip install <package-name>' command from PyPI"
    else:
        claim = f"The README does NOT show that this repository can be installed with a single 'pip install <package-name>' command from PyPI"

    await evaluator.verify(
        claim=claim,
        node=pip_claim_node,
        sources=[readme_url] + [repo_info.github_url],
        additional_instruction="A single pip install command means 'pip install package-name' that installs from PyPI. Commands like 'pip install -e .' or 'pip install git+...' or installation requiring additional steps do NOT count."
    )

    # PyPI URL verification (parallel with two critical checks)
    pypi_verification_node = evaluator.add_parallel(
        id=f"repo_{repo_index}_pypi_verification",
        desc=f"PyPI URL verification",
        parent=pip_info_node,
        critical=True,
    )

    # Check PyPI URL existence based on can_pip_install
    pypi_exists_check = evaluator.add_custom_node(
        result=(
            # If can_pip_install is True, must have PyPI URL
                (repo_info.can_pip_install and bool(repo_info.pypi_url)) or
                # If can_pip_install is False, PyPI URL not required
                (not repo_info.can_pip_install)
        ),
        id=f"repo_{repo_index}_pypi_url_exists_correctly",
        desc=f"PyPI URL existence matches pip install claim (can_pip_install={repo_info.can_pip_install})",
        parent=pypi_verification_node,
        critical=True,
    )

    # Verify PyPI URL validity if can_pip_install is True
    if repo_info.can_pip_install:
        pypi_url_valid_node = evaluator.add_leaf(
            id=f"repo_{repo_index}_pypi_url_valid",
            desc=f"PyPI URL is valid and corresponds to the repository",
            parent=pypi_verification_node,
            critical=True,
        )

        await evaluator.verify(
            claim=f"The PyPI page at {repo_info.pypi_url} exists and is for a package related to '{repo_info.name}'",
            node=pypi_url_valid_node,
            sources=repo_info.pypi_url,
            additional_instruction="Verify the PyPI page exists and the package name corresponds to the repository. Allow for significant naming variations as PyPI package names often differ from GitHub repository names."
        )
    else:
        # If can_pip_install is False, automatically pass this check
        pypi_url_pass_node = evaluator.add_custom_node(
            result=True,
            id=f"repo_{repo_index}_pypi_url_not_needed",
            desc="PyPI URL verification not needed (can_pip_install=False)",
            parent=pypi_verification_node,
            critical=True,
        )


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
    """Main evaluation function for RAG repository task"""

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

    # Extract repository information from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_repositories(),
        template_class=ExtractedRepositories,
        extraction_name="extracted_repositories",
    )

    # Extract ground truth top 10 repositories
    ground_truth = await evaluator.extract(
        prompt=prompt_extract_top_repos(),
        template_class=TopRepositoriesList,
        extraction_name="ground_truth_top_10",
        source=GROUND_TRUTH_URL,
    )

    # Create ground truth string for verification
    ground_truth_repos_string = ", ".join(ground_truth.top_10_repo_full_names)

    # Add ground truth info
    evaluator.add_ground_truth({
        "top_10_repositories": ground_truth.top_10_repo_full_names,
        "source": GROUND_TRUTH_URL
    })

    # Check if we have any repositories
    has_repos = evaluator.add_custom_node(
        result=len(extracted.repositories) > 0,
        id="has_repositories",
        desc="Answer contains at least one repository",
        parent=root,
        critical=True,  # Critical - no repos means complete failure
    )

    # Get first 10 repositories (pad with empty if needed)
    repositories = extracted.repositories[:10]  # Take first 10 if more provided
    while len(repositories) < 10:
        repositories.append(RepositoryInfo())  # Add empty repos for missing ones

    # Verify each repository
    for i, repo in enumerate(repositories):
        await verify_repository(
            evaluator=evaluator,
            parent_node=root,
            repo_info=repo,
            repo_index=i,
            ground_truth_repos_string=ground_truth_repos_string,
        )

    # Add custom info about completeness
    evaluator.add_custom_info({
        "repositories_provided": len(extracted.repositories),
        "repositories_expected": 10,
        "completeness_ratio": min(len(extracted.repositories), 10) / 10
    }, "answer_completeness")

    return evaluator.get_summary()