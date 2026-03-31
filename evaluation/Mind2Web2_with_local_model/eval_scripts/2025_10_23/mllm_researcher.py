import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mllm_researcher"
TASK_DESCRIPTION = """
Find a researcher who is currently affiliated with UC Berkeley or received their PhD from UC Berkeley, and who has a paper related to vision and language models that was accepted at ICLR 2023. Provide the researcher's name, a description of their UC Berkeley affiliation (such as current position at Berkeley or PhD obtained there), a link to the paper's page on OpenReview, and the final recommendation ratings given by each reviewer for the paper.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResearcherInfo(BaseModel):
    name: Optional[str] 
    berkeley_affiliation_description: Optional[str]   # Full sentence describing the affiliation


class PaperInfo(BaseModel):
    title: Optional[str]
    openreview_url: Optional[str]
    reviewer_ratings_description: Optional[str]  # Full description of all reviewer ratings


class ExtractedAnswer(BaseModel):
    researcher: Optional[ResearcherInfo]
    paper: Optional[PaperInfo]
    all_urls: List[str] = Field(default_factory=list)  # All URLs mentioned in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_researcher_and_paper():
    return """
    Extract the following information from the answer:

    1. Researcher Information:
       - name: The researcher's name (just the name)
       - berkeley_affiliation_description: The complete sentence or description explaining their UC Berkeley affiliation (e.g., "John Smith is currently a Professor at UC Berkeley" or "Jane Doe received her PhD from UC Berkeley in 2018")

    2. Paper Information:
       - title: The paper title (if mentioned)
       - openreview_url: The OpenReview URL for the paper
       - reviewer_ratings_description: The complete description of all reviewer ratings mentioned in the answer (e.g., "The reviewers gave ratings of 6, 7, 8, and 6" or "Final recommendation ratings: Accept (6), Accept (7), Accept (8), Weak Accept (5)")

    3. all_urls: Extract ALL URLs mentioned anywhere in the answer

    Return the information in the specified JSON structure. If any information is missing, set it to null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_researcher_information(
        parent_node,
        info: ExtractedAnswer,
        evaluator: Evaluator,
) -> None:
    """
    Verify that the researcher's name and UC Berkeley affiliation are provided and supported by URLs.
    Returns True if verification passed, False otherwise.
    """
    researcher_node = evaluator.add_parallel(
        id="researcher_information",
        desc="The answer provides a researcher name and UC Berkeley affiliation description that are supported by the provided URLs.",
        critical=True,
        parent=parent_node
    )

    # Step 1: Check if basic information is provided
    basic_info_valid = (info.researcher and
                        info.researcher.name and
                        info.researcher.berkeley_affiliation_description and
                        info.all_urls)
    
    basic_info_node = evaluator.add_custom_node(
        result=basic_info_valid,
        id="researcher_basic_info",
        desc="The answer provides both researcher name and UC Berkeley affiliation description, and URLs that can be used for verification.",
        parent=researcher_node,
        critical=True
    )

    # Step 2: Verify researcher information using URLs
    url_verification_node = evaluator.add_leaf(
        id="researcher_url_verification",
        desc="The researcher information is supported by the provided URLs.",
        parent=researcher_node,
        critical=True
    )

    # Create comprehensive claim for verification
    researcher_claim = (
        f"The researcher {info.researcher.name} has the following UC Berkeley affiliation: "
        f"{info.researcher.berkeley_affiliation_description}. This indicates that they are "
        f"either currently affiliated with UC Berkeley or received their PhD from UC Berkeley."
    )

    # Updated additional instruction to be more flexible with terminology variations
    additional_instruction = (
        "Verify that the researcher has a UC Berkeley affiliation as described. "
        "Allow for reasonable variations in terminology and phrasing - for example, "
        "'co-founding' vs 'co-directing', 'faculty member' vs 'professor', "
        "'Computer Science Division' vs 'EECS department', etc. "
        "The key requirement is that the person has a clear current affiliation with UC Berkeley "
        "or obtained their PhD from UC Berkeley. Focus on the substance of the affiliation "
        "rather than exact wording matches."
    )

    # Verify using all available URLs
    verified = await evaluator.verify(
        claim=researcher_claim,
        node=url_verification_node,
        sources=info.all_urls,
        additional_instruction=additional_instruction
    )


async def verify_openreview_paper_basic(
        parent_node,
        info: ExtractedAnswer,
        evaluator: Evaluator,
) -> None:
    """
    Verify that a valid OpenReview URL is provided and corresponds to the mentioned paper title.
    Returns True if verification passed, False otherwise.
    """

    # Check if paper information is provided
    paper_info_valid = (info.paper and
                        info.paper.openreview_url and 'openreview' in info.paper.openreview_url)
    paper_info_node = evaluator.add_custom_node(
        result=paper_info_valid,
        id="paper_info_provided",
        desc="The answer provides both paper title and OpenReview URL.",
        parent=parent_node,
        critical=True
    )


async def verify_paper_comprehensive(
        parent_node,
        info: ExtractedAnswer,
        evaluator: Evaluator,
) -> None:
    """
    Verify that the paper is ICLR 2023, vision-language related, and authored by the researcher.
    Returns True if verification passed, False otherwise.
    """
    node = evaluator.add_leaf(
        id="paper_comprehensive",
        desc="The paper was accepted at ICLR 2023, is related to vision and language models, and the named researcher is an author.",
        parent=parent_node,
        critical=True
    )

    # Create comprehensive claim that bundles all requirements
    paper_comprehensive_claim = (
        f"This page shows a paper and it meets all of the following requirements: "
        f"(1) it was accepted at ICLR 2023, "
        f"(2) it is related to vision and language models, and "
        f"(3) {info.researcher.name} is listed as one of the authors."
    )

    # Additional instruction to help the verifier
    additional_instruction = (
        "Please verify all three requirements: (1) ICLR 2023 acceptance, "
        "(2) vision-language topic (including multimodal models, image captioning, visual QA, etc.), "
        "and (3) the researcher's authorship. All three must be true for the claim to be supported."
    )

    verified = await evaluator.verify(
        claim=paper_comprehensive_claim,
        node=node,
        sources=info.paper.openreview_url,
        additional_instruction=additional_instruction
    )


async def verify_reviewer_ratings(
        parent_node,
        info: ExtractedAnswer,
        evaluator: Evaluator
) -> None:
    """
    Verify that the reviewer ratings mentioned in the answer match those on the OpenReview page.
    Returns True if verification passed, False otherwise.
    """
    ratings_node = evaluator.add_sequential(
        id="reviewer_ratings",
        desc="The reviewer ratings mentioned in the answer match the actual ratings on the OpenReview page.",
        critical=True,
        parent=parent_node
    )

    # Step 1: Check if reviewer ratings description is provided
    ratings_provided = (info.paper and
                        info.paper.reviewer_ratings_description and
                        info.paper.reviewer_ratings_description.strip())
    ratings_provided_node = evaluator.add_custom_node(
        result=ratings_provided,
        id="ratings_description_provided",
        desc="The answer provides a description of reviewer ratings.",
        parent=ratings_node,
        critical=True
    )

    # Step 2: Verify the ratings match OpenReview page
    ratings_verification_node = evaluator.add_leaf(
        id="ratings_url_verification",
        desc="The mentioned reviewer ratings match those on the OpenReview page.",
        parent=ratings_node,
        critical=True
    )

    # Create claim about the reviewer ratings
    ratings_claim = (
        f"The reviewer ratings for the paper at {info.paper.openreview_url} are: "
        f"{info.paper.reviewer_ratings_description}"
    )

    verified = await evaluator.verify(
        claim=ratings_claim,
        node=ratings_verification_node,
        sources=info.paper.openreview_url,
        additional_instruction="Check if the mentioned reviewer ratings match the actual final recommendation ratings shown on the OpenReview page."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: Any,  # Using Any instead of openai.AsyncAzureOpenAI for flexibility
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate whether the answer correctly identifies a UC Berkeley researcher with
    a vision-language paper at ICLR 2023, and provides all the required information.
    """
    # Initialize evaluator
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

    # Extract structured information from the answer
    logger.info("Extracting researcher and paper information...")
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_researcher_and_paper(),
        template_class=ExtractedAnswer,
        extraction_name="researcher_and_paper_extraction"
    )

    # Build verification tree with sequential structure

    # ========== STEP 1: RESEARCHER VERIFICATION ==========
    await verify_researcher_information(
        root,
        extracted_info,
        evaluator
    )

    # ========== STEP 2: PAPER VERIFICATION ==========
    # Create a parent node for all paper-related verifications
    paper_verification_node = evaluator.add_parallel(
        id="paper_verification",
        desc="The answer provides a valid ICLR 2023 vision-language paper with correct details.",
        critical=True,
        parent=root
    )

    # Step 2.1: Basic OpenReview verification
    await verify_openreview_paper_basic(
        paper_verification_node,
        extracted_info,
        evaluator,
    )

    # Step 2.2: Comprehensive paper verification
    await verify_paper_comprehensive(
        paper_verification_node,
        extracted_info,
        evaluator,
    )

    # Step 2.3: Reviewer ratings verification
    await verify_reviewer_ratings(
        paper_verification_node,
        extracted_info,
        evaluator,
    )

    # Return structured result using evaluator's summary
    return evaluator.get_summary()