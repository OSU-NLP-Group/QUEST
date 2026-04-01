import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "paper_collaboration"
TASK_DESCRIPTION = """
I'm interested in the paper "ASDOT: Any-shot data-to-text generation with pretrained language models." Provide the arxiv link and the name of the first author. Identify their PhD advisor, providing the advisor's homepage and stated primary research interests. Then, find three other researchers (excluding the PhD advisor) who have co-authored at least three papers with this first author. For each researcher, list their full name and three co-authored paper titles, including the arxiv link for each paper.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {
    "arxiv_link": "https://arxiv.org/abs/2210.04325",
    "first_author": "Jiannan Xiang",
    "phd_advisor": "Zhiting Hu"
}


# Step 1: Basic paper info
class BasicPaperInfo(BaseModel):
    """Basic information about the ASDOT paper"""
    arxiv_link: Optional[str] = Field(default=None, description="ArXiv link of the ASDOT paper")
    first_author: Optional[str] = Field(default=None, description="Name of the first author")


# Step 2: Advisor info
class AdvisorInfo(BaseModel):
    """PhD advisor information"""
    advisor_name: Optional[str] = Field(default=None, description="Name of the PhD advisor")
    advisor_homepage: Optional[str] = Field(default=None, description="Homepage URL of the PhD advisor")
    advisor_research_interests: Optional[str] = Field(
        default=None,
        description="Stated primary research interests of the PhD advisor"
    )


# Step 3: Collaborator names
class CollaboratorNames(BaseModel):
    """Names of collaborators"""
    collaborator_names: List[str] = Field(
        default_factory=list,
        description="List of collaborator names (excluding PhD advisor)"
    )


# Step 4: Papers for a single collaborator
class CollaboratorPapers(BaseModel):
    """Papers co-authored with a specific collaborator"""
    paper_titles: List[str] = Field(
        default_factory=list,
        description="List of paper titles"
    )
    arxiv_links: List[str] = Field(
        default_factory=list,
        description="List of arxiv links corresponding to the papers"
    )


def prompt_extract_basic_paper_info() -> str:
    """Extract basic paper information"""
    return """
    Extract basic information about the ASDOT paper from the answer.

    Look for:
    1. arxiv_link: The ArXiv link for the paper "ASDOT: Any-shot data-to-text generation with pretrained language models"
    2. first_author: The name of the first author of the ASDOT paper

    Extract information exactly as it appears in the text.
    For URLs, ensure they are complete and valid (including http:// or https:// prefix).
    """


def prompt_extract_advisor_info() -> str:
    """Extract PhD advisor information"""
    return """
    Extract information about the PhD advisor from the answer.

    Look for:
    1. advisor_name: The name of the first author's PhD advisor
    2. advisor_homepage: The homepage URL of the PhD advisor
    3. advisor_research_interests: The stated primary research interests of the PhD advisor (extract as a single string)

    Extract information exactly as it appears in the text.
    For URLs, ensure they are complete and valid (including http:// or https:// prefix).
    """


def prompt_extract_collaborator_names() -> str:
    """Extract collaborator names"""
    return """
    Extract the names of researchers who have co-authored at least 3 papers with the first author.

    Look for:
    - collaborator_names: A list of researcher names (excluding the PhD advisor)

    Extract only the names, not their papers. Extract names exactly as they appear in the text.
    """


def prompt_extract_collaborator_papers(collaborator_name: str) -> str:
    """Extract papers for a specific collaborator"""
    return f"""
    Extract information about papers co-authored between the first author and {collaborator_name}.

    Look for:
    1. paper_titles: List of paper titles that were co-authored
    2. arxiv_links: List of ArXiv links corresponding to these papers. When doing the extraction, ensure that the order of the links matches the order of the titles. And, plz correctly extract the links, for example, if the link is "https://arxiv.org/abs/1234.5678", then the extracted link should be "https://arxiv.org/abs/1234.5678" and not "arxiv.org/abs/1234.5678" or "https://arxiv.org/abs/1234.5678.pdf" or with any unnecessary or false suffix.

    Extract the titles and links in the same order so they correspond to each other.
    Extract information exactly as it appears in the text.
    For URLs, ensure they are complete and valid (including http:// or https:// prefix).
    """


async def verify_paper_and_advisor(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        paper_info: BasicPaperInfo,
        advisor_info: AdvisorInfo,
) -> None:
    """Verify paper information and advisor information"""
    # Create sequential node for paper and advisor
    paper_advisor_node = evaluator.add_sequential(
        id="paper_and_advisor",
        desc="Paper information and PhD advisor information",
        parent=parent_node,
        critical=False  # Non-critical for partial credit
    )

    # Verify paper info
    await verify_paper_info(evaluator, paper_advisor_node, paper_info)

    # Verify advisor info
    await verify_advisor_info(evaluator, paper_advisor_node, advisor_info)


async def verify_paper_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        info: BasicPaperInfo,
) -> None:
    """Verify the basic paper information"""
    paper_node = evaluator.add_parallel(
        id="paper_info",
        desc="ASDOT paper basic information",
        parent=parent_node,
        critical=False  # Critical within the sequential parent
    )

    # Combined existence check
    existence_node = evaluator.add_custom_node(
        result=bool(info.arxiv_link and info.first_author),
        id="paper_info_exists",
        desc="ArXiv link and first author name are provided",
        parent=paper_node,
        critical=True
    )

    # Verify ArXiv link
    if not info.arxiv_link:
        info.arxiv_link=""
        # Check if arxiv ID is in the link
    arxiv_id = "2210.04325"
    if arxiv_id in info.arxiv_link and "arxiv" in info.arxiv_link.lower():
        # Direct validation
        arxiv_correct = evaluator.add_custom_node(
            result=True,
            id="arxiv_link_correct",
            desc="ArXiv link contains correct ID and arxiv domain",
            parent=paper_node,
            critical=True
        )
    else:
        # Verify by URL
        arxiv_correct = evaluator.add_leaf(
            id="arxiv_link_correct",
            desc="ArXiv link leads to the correct ASDOT paper",
            parent=paper_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The URL '{info.arxiv_link}' leads to the paper 'ASDOT: Any-shot data-to-text generation with pretrained language models'",
            node=arxiv_correct,
            sources=info.arxiv_link,
            additional_instruction="Verify this is the correct ASDOT paper by checking the title on the page"
        )

    # Verify first author against ground truth
    # if info.first_author:
    author_correct = evaluator.add_custom_node(
        result=(info.first_author.lower().strip() == GROUND_TRUTH['first_author'].lower().strip()),
        id="first_author_correct",
        desc=f"First author name matches ground truth: {info.first_author} vs {GROUND_TRUTH['first_author']}",
        parent=paper_node,
        critical=True
    )


async def verify_advisor_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        info: AdvisorInfo,
) -> None:
    """Verify PhD advisor information"""
    advisor_node = evaluator.add_parallel(
        id="advisor_info",
        desc="PhD advisor information",
        parent=parent_node,
        critical=False  # Critical within the sequential parent
    )

    # Combined existence check
    existence_node = evaluator.add_custom_node(
        result=bool(info.advisor_name and info.advisor_homepage and info.advisor_research_interests),
        id="advisor_info_exists",
        desc="All advisor information (name, homepage, research interests) is provided",
        parent=advisor_node,
        critical=True
    )

    # Verify advisor name against ground truth
    if not info.advisor_name:
        info.advisor_name=""

    name_correct = evaluator.add_leaf(
        id="advisor_name_correct",
        desc=f"Advisor name matches ground truth: {info.advisor_name} vs {GROUND_TRUTH['phd_advisor']}",
        parent=advisor_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The name '{info.advisor_name}' matches the ground truth advisor name '{GROUND_TRUTH['phd_advisor']}'",
        node=name_correct,
    )



    # Verify advisor homepage
    # if info.advisor_homepage:
    homepage_valid = evaluator.add_leaf(
        id="advisor_homepage_valid",
        desc="Advisor homepage URL is valid",
        parent=advisor_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The URL '{info.advisor_homepage}' is a valid homepage for {info.advisor_name or 'the PhD advisor'}",
        node=homepage_valid,
        sources=info.advisor_homepage,
        additional_instruction="Verify this is a personal/faculty homepage, not just any webpage"
    )

    # Verify research interests
    # if info.advisor_research_interests and info.advisor_homepage:
    interests_valid = evaluator.add_leaf(
        id="advisor_interests_valid",
        desc="Research interests reasonably match homepage information",
        parent=advisor_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The research interests '{info.advisor_research_interests}' reasonably match the information on the advisor's homepage",
        node=interests_valid,
        sources=info.advisor_homepage,
        additional_instruction="""Verify that the stated research interests:
1. Are NOT generic terms like just 'AI' or 'AI Agents' without specifics
2. Match the general direction shown on the homepage (doesn't need to cover everything)
3. Don't contain obviously fabricated interests that contradict the homepage
4. Can include reasonable inferences from papers/projects but should be grounded in homepage content
The interests should reflect the main research areas accurately, though minor variations or different phrasings are acceptable."""
    )


async def verify_single_collaborator(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        collaborator_name: Optional[str],
        papers: CollaboratorPapers,
        collaborator_index: int,
        first_author_name: str,
) -> None:
    """Verify a single collaborator and their papers"""
    collab_node = evaluator.add_parallel(
        id=f"collaborator_{collaborator_index}",
        desc=f"Collaborator {collaborator_index + 1}: {collaborator_name or 'Not provided'}",
        parent=parent_node,
        critical=False  # Non-critical for partial credit
    )

    # Check name exists
    name_exists = evaluator.add_custom_node(
        result=bool(collaborator_name),
        id=f"collaborator_{collaborator_index}_name_exists",
        desc=f"Collaborator {collaborator_index + 1} name is provided",
        parent=collab_node,
        critical=True
    )

    # Verify papers
    papers_node = evaluator.add_parallel(
        id=f"collaborator_{collaborator_index}_papers",
        desc=f"Papers co-authored with collaborator {collaborator_index + 1}",
        parent=collab_node,
        critical=True
    )

    # Verify exactly 3 papers (or fewer if less provided)
    num_papers_to_verify = min(3, len(papers.paper_titles), len(papers.arxiv_links))

    for paper_idx in range(3):
        if paper_idx < num_papers_to_verify:
            title = papers.paper_titles[paper_idx] if paper_idx < len(papers.paper_titles) else None
            link = papers.arxiv_links[paper_idx] if paper_idx < len(papers.arxiv_links) else None

            if title and link:
                paper_node = evaluator.add_leaf(
                    id=f"collaborator_{collaborator_index}_paper_{paper_idx}",
                    desc=f"Paper {paper_idx + 1}: {title[:50] + '...' if len(title) > 50 else title}",
                    parent=papers_node,
                    critical=True
                )

                await evaluator.verify(
                    claim=f"The paper titled '{title}' at {link} is co-authored by both {first_author_name} and {collaborator_name}",
                    node=paper_node,
                    sources=link,
                    additional_instruction="""Check the author list to confirm both authors are listed. 
Allow reasonable name variations. 
For the title, allow variations like:
- Case differences (uppercase/lowercase)
- Minor abbreviations or omissions
- Core title matches even if subtitles differ
But the title should not be completely different."""
                )
            else:
                # Missing title or link
                evaluator.add_custom_node(
                    result=False,
                    id=f"collaborator_{collaborator_index}_paper_{paper_idx}_missing",
                    desc=f"Paper {paper_idx + 1} information incomplete",
                    parent=papers_node,
                    critical=True
                )
        else:
            # Create placeholder for missing paper
            evaluator.add_custom_node(
                result=False,
                id=f"collaborator_{collaborator_index}_paper_{paper_idx}_not_provided",
                desc=f"Paper {paper_idx + 1} not provided",
                parent=papers_node,
                critical=True
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
    Main evaluation function for the paper collaboration task.

    Evaluates:
    1. Paper information (arxiv link and first author)
    2. PhD advisor details (name, homepage, research interests)
    3. Three collaborators with their co-authored papers
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Changed to SEQUENTIAL
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

    # -------- 2. Step-by-step extraction -------------------------- #

    # Step 1: Extract basic paper info
    paper_info = await evaluator.extract(
        prompt=prompt_extract_basic_paper_info(),
        template_class=BasicPaperInfo,
        extraction_name="basic_paper_info",
    )

    # Step 2: Extract advisor info
    advisor_info = await evaluator.extract(
        prompt=prompt_extract_advisor_info(),
        template_class=AdvisorInfo,
        extraction_name="advisor_info",
    )

    # Step 3: Extract collaborator names
    collaborator_names = await evaluator.extract(
        prompt=prompt_extract_collaborator_names(),
        template_class=CollaboratorNames,
        extraction_name="collaborator_names",
    )

    # Step 4: Extract papers for each collaborator
    all_collaborator_papers = []
    for i, name in enumerate(collaborator_names.collaborator_names[:3]):  # Only first 3
        papers = await evaluator.extract(
            prompt=prompt_extract_collaborator_papers(name),
            template_class=CollaboratorPapers,
            extraction_name=f"collaborator_{i}_papers",
        )
        all_collaborator_papers.append((name, papers))

    # Add ground truth information
    evaluator.add_ground_truth(GROUND_TRUTH, "paper_ground_truth")

    # -------- 3. Build verification tree -------------------------- #

    # Verify paper and advisor info (first sequential node)
    await verify_paper_and_advisor(evaluator, root, paper_info, advisor_info)

    # Verify collaborators (second sequential node)
    collaborators_node = evaluator.add_parallel(
        id="collaborators",
        desc="Three collaborators with co-authored papers",
        parent=root,
        critical=False  # Non-critical for partial credit
    )

    first_author_name = paper_info.first_author or GROUND_TRUTH['first_author']

    # Process exactly 3 collaborators
    for i in range(3):
        if i < len(all_collaborator_papers):
            name, papers = all_collaborator_papers[i]
            await verify_single_collaborator(
                evaluator,
                collaborators_node,
                name,
                papers,
                i,
                first_author_name
            )
        else:
            # Create placeholder for missing collaborator
            await verify_single_collaborator(
                evaluator,
                collaborators_node,
                None,
                CollaboratorPapers(),
                i,
                first_author_name
            )

    # -------- 4. Return evaluation results ------------------------ #
    return evaluator.get_summary()