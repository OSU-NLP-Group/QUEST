import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gcg_followup"
TASK_DESCRIPTION = """
I am applying for a PhD program with a focus on large language model (LLM) safety. A notable paper in this field is *"Universal and Transferable Adversarial Attacks on Aligned Language Models,"* which introduced the GCG algorithm and has inspired numerous follow-up studies. I recall that one of these follow-up works identified certain limitations in the GCG loss function and trained an LLM to generate GCG-like suffixes. Could you help me find that paper? I am particularly interested in pursuing a PhD under the supervision of the last author of that follow-up paper. Could you identify who the last author is, find his/her homepage where their information is directly displayed, and list all other safety/security-related papers (with direct links) that this professor published in 2024?
"""

JUDGE_MODEL = "o4-mini"

# Ground truth constants
GROUND_TRUTH_PAPER = "Amplegcg: Learning a universal and transferable generative model of adversarial suffixes for jailbreaking both open and closed llms"
GROUND_TRUTH_AUTHOR = "Huan Sun"

# Must-have papers by Huan Sun in 2024
MUST_HAVE_PAPERS = [
    "Advweb: Controllable black-box attacks on vlm-powered web agents",
    "Autodan-turbo: A lifelong agent for strategy self-exploration to jailbreak llms",
    "EIA: Environmental injection attack on generalist web agents for privacy leakage",
    "A trembling house of cards? mapping adversarial attacks against language agents",
    "Combating security and privacy issues in the era of large language models",
    "AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts"
]

# Optional papers (not required for evaluation)
OPTIONAL_PAPERS = [
    "WebOlympus: An Open Platform for Web Agents on Live Websites",
    "Amplegcg: Learning a universal and transferable generative model of adversarial suffixes for jailbreaking both open and closed llms",
    "How Trustworthy are Open-Source LLMs? An Assessment under Malicious Demonstrations Shows their Vulnerabilities"
]


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    title: Optional[str]
    last_author: Optional[str]
    paper_urls: List[str] = Field(default_factory=list)  # Added to store paper URLs directly


class HomepageInfo(BaseModel):
    homepage_urls: List[str] = Field(default_factory=list)


class PaperEntry(BaseModel):
    title: str
    urls: List[str] = Field(default_factory=list)


class AuthorPapers(BaseModel):
    papers: List[PaperEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_info() -> str:
    return """
    Extract the following information from the answer:
    1. The title of the follow-up paper to the GCG algorithm that identified limitations in the GCG loss function and trained an LLM to generate GCG-like suffixes.
    2. The last author of this paper.
    3. All URLs provided in the answer that link directly to this follow-up paper.

    Return null for title or last_author if they cannot be clearly identified.
    Return an empty list for paper_urls if no URLs are provided for the paper.
    """


def prompt_extract_homepage() -> str:
    return """
    Extract all homepage URLs of the last author (professor) mentioned in the answer.
    These should be direct URLs to their personal/academic homepage, not general university pages or paper repositories.

    Return an empty list if no homepage URLs are clearly provided in the answer.
    """

def prompt_extract_papers() -> str:
    optional_paper_list = "\n".join(OPTIONAL_PAPERS)
    return f"""
    Extract all papers mentioned in the answer text that are provided as safety/security-related papers published by the professor (the last author) in 2024.

    Note: Do NOT include these papers in your extraction (they will be handled separately) (simply skip/ignore them):

    {optional_paper_list}

    For each paper, extract:
    1. The paper title
    2. All URL links to the paper (if provided), returned as a list of URLs

    Return an empty list if no papers are mentioned.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_paper(
        evaluator: Evaluator,
        parent_node,
        paper_info: PaperInfo,
) -> None:
    """
    Verify the paper title and URL in a combined node.
    """
    # Create parent node for paper verification
    paper_parent_node = evaluator.add_parallel(
        id="paper_verification",
        desc="Paper Verification: AmpleGCG paper title and URL",
        parent=parent_node,
        critical=False,
    )

    # 1. Combined existence check for paper info
    paper_exists_node = evaluator.add_custom_node(
        result=(paper_info.title is not None and paper_info.title.strip() != "") and bool(paper_info.paper_urls),
        id="paper_info_exists",
        desc="Check if both paper title and URLs were provided in the answer",
        parent=paper_parent_node,
        critical=True
    )

    # 2. Verify paper title
    title_verification_node = evaluator.add_leaf(
        id="paper_title_verification",
        desc="Verification of the correct paper title (AmpleGCG) being identified",
        parent=paper_parent_node,
        critical=True,
    )

    claim = f"The paper title '{paper_info.title}' matches or is equivalent to the expected paper title '{GROUND_TRUTH_PAPER}', or matches or contains the commonly used abbreviation 'AmpleGCG'. It's okay to omit part of the title here."
    await evaluator.verify(
        claim=claim,
        node=title_verification_node,
        additional_instruction="Check if the titles are semantically equivalent, allowing for minor variations in capitalization, punctuation, or short words."
    )

    # 3. Verify paper URL
    url_verification_node = evaluator.add_leaf(
        id="paper_url_verification",
        desc="Verification that a valid URL was provided for the AmpleGCG paper",
        parent=paper_parent_node,
        critical=True,
    )

    claim = f"The URL leads to a page containing or directly referencing the AmpleGCG paper."
    await evaluator.verify(
        claim=claim,
        node=url_verification_node,
        sources=paper_info.paper_urls,
        additional_instruction="Verify that this URL links to the paper, either to the paper itself, its abstract page, or a repository page clearly featuring this paper. The title should be clearly visible or referenced on the page."
    )


async def verify_author(
        evaluator: Evaluator,
        parent_node,
        paper_info: PaperInfo,
        homepage_info: HomepageInfo,
) -> None:
    """
    Verify the author name and homepage in a combined node.
    """
    # Create parent node for author verification
    author_parent_node = evaluator.add_parallel(
        id="author_verification",
        desc="Author Verification: Huan Sun's identity and homepage",
        parent=parent_node,
        critical=False,
    )

    # 1. Combined existence check for author info
    author_exists_node = evaluator.add_custom_node(
        result=(paper_info.last_author is not None and paper_info.last_author.strip() != "") and bool(homepage_info.homepage_urls),
        id="author_info_exists",
        desc="Check if both author name and homepage URLs were provided in the answer",
        parent=author_parent_node,
        critical=True
    )

    # 2. Verify author name
    name_verification_node = evaluator.add_leaf(
        id="author_name_verification",
        desc="Verification of the correct last author (Huan Sun) being identified",
        parent=author_parent_node,
        critical=True,
    )

    claim = f"The author name '{paper_info.last_author}' is exactly 'Huan Sun' or an equivalent form."
    await evaluator.verify(
        claim=claim,
        node=name_verification_node,
        additional_instruction="This is used to check whether an answer found the correct person. So, for the name verification, the name must be Huan Sun. But minor variations in capitalization or inclusion of middle initials are acceptable."
    )

    # 3. Verify homepage
    homepage_verification_node = evaluator.add_leaf(
        id="homepage_verification",
        desc="Verification that a valid homepage URL for Huan Sun was provided",
        parent=author_parent_node,
        critical=True,
    )

    claim = "The URL is a valid homepage for Huan Sun, showing her direct personal/academic information."
    await evaluator.verify(
        claim=claim,
        node=homepage_verification_node,
        sources=homepage_info.homepage_urls,
        additional_instruction="Verify that this is a personal/academic homepage specifically for Huan Sun, not just a general university page, publication repository, or social media profile. The page should directly display information about Huan Sun, such as her position, research interests, publications, etc."
    )


async def verify_specific_paper(
        evaluator: Evaluator,
        parent_node,
        paper_index: int,
        must_have_paper: str,
        author_papers: AuthorPapers,
) -> None:
    """
    Verify if a specific must-have paper is mentioned in the answer.
    """
    paper_node = evaluator.add_parallel(
        id=f"paper_{paper_index}_verification",
        desc=f"Verification of paper {paper_index}: '{must_have_paper}'",
        parent=parent_node,
        critical=False,  # Not critical to allow partial credit
    )

    # Check if the paper is in the extracted papers
    matching_papers = []
    for paper in author_papers.papers:
        claim = f"The paper title '{paper.title}' matches the paper '{must_have_paper}'."
        is_match = await evaluator.verify(
            claim=claim,
            node=None,  # Don't assign to any node, just get the result
            additional_instruction="Check if the titles are equivalent, allowing for minor variations in capitalization, punctuation, or short words.",
        )
        if is_match:
            matching_papers.append(paper)
            break

    # 1. Combined existence check for paper and its URLs
    paper_exists_node = evaluator.add_custom_node(
        result=bool(matching_papers) and bool(matching_papers[0].urls if matching_papers else False),
        id=f"paper_{paper_index}_info_exists",
        desc=f"Check if paper {paper_index} '{must_have_paper}' was found with URLs",
        parent=paper_node,
        critical=True
    )

    # 2. Verify URL
    url_verification_node = evaluator.add_leaf(
        id=f"paper_{paper_index}_url_verification",
        desc=f"Verification that a valid URL was provided for paper {paper_index}: '{must_have_paper}'",
        parent=paper_node,
        critical=True,
    )

    matched_paper = matching_papers[0] if matching_papers else None
    paper_sources = matched_paper.urls if matched_paper else None
    claim = f"The URL leads to a page containing or directly referencing the paper titled '{must_have_paper}'."
    await evaluator.verify(
        claim=claim,
        node=url_verification_node,
        sources=paper_sources,
        additional_instruction="Verify that this URL links to the paper, either to the paper itself, its abstract page, or a repository page clearly featuring this paper. The title should be clearly visible or referenced on the page.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with sequential strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract structured info from the answer ---------------- #
    # Extract paper title, last author, and paper URLs
    paper_info = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=PaperInfo,
        extraction_name="paper_info"
    )

    # Extract homepage URLs
    homepage_info = await evaluator.extract(
        prompt=prompt_extract_homepage(),
        template_class=HomepageInfo,
        extraction_name="homepage_info"
    )

    # Extract papers published by the author in 2024
    author_papers = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=AuthorPapers,
        extraction_name="author_papers"
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Step 1: Paper, Author, and Homepage verification (50% of score)
    first_task_node = evaluator.add_sequential(
        id="first_task",
        desc="First Task: Paper, Author and Homepage Verification",
        critical=False,  # Non-critical to allow partial credit
    )

    # Verify paper (title and URL) - now using only paper_info
    await verify_paper(evaluator, first_task_node, paper_info)

    # Verify author (name and homepage)
    await verify_author(evaluator, first_task_node, paper_info, homepage_info)

    # Step 2: Papers verification (50% of score)
    second_task_node = evaluator.add_parallel(
        id="second_task",
        desc="Second Task: Verification of Huan Sun's 2024 safety/security papers",
        critical=False,  # Non-critical to allow partial credit
    )

    # Verify each must-have paper specifically
    for i, must_have_paper in enumerate(MUST_HAVE_PAPERS):
        await verify_specific_paper(
            evaluator,
            second_task_node,
            i + 1,
            must_have_paper,
            author_papers,
        )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()