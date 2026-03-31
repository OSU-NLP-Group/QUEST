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
TASK_ID = "perplexity_failure"
TASK_DESCRIPTION = """
I'm looking for real-world examples of failures involving Perplexity AI in the context of AI-powered search. Specifically, I am interested in cases where Perplexity AI's search products (for example, their Pro Search, or Deep Research) produced hallucinated answers—such as fabricated URLs or made-up information. Please find three links, each of which should include at least one concrete example of such a failure (for example, the user input and the AI's response, or at least a concrete description of what happened). The sources should not merely mention general problems with Perplexity AI, but must describe specific instances where it provided hallucinated responses.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PerplexityFailureLink(BaseModel):
    """Represents a single link to a Perplexity AI failure example."""
    url: Optional[str] = None

class PerplexityFailureInfo(BaseModel):
    """Extracted information about Perplexity AI failures."""
    failure_links: List[PerplexityFailureLink] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_failure_links() -> str:
    return """
    Extract all links (URLs) mentioned in the answer that point to examples of Perplexity AI hallucination failures.
    
    For each link, extract the URL itself
    
    Return all links in the order they appear in the answer. If no links are provided in the answer, return an empty list.
    If the URL is malformed or missing a protocol (e.g., "example.com" instead of "https://example.com"), prepend "http://" to make it valid.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_link_validity(
    evaluator: Evaluator,
    parent_node,
    link: Optional[PerplexityFailureLink],
    link_index: int,
) -> None:
    """
    Verify that a specific link contains a concrete example of a Perplexity AI hallucination failure.
    """
    # Create parent node for link verification
    link_parent = evaluator.add_parallel(
        id=f"link_{link_index}_verification",
        desc=f"Link {link_index} Verification",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )
    
    # 1. Check if link exists
    link_exists = evaluator.add_custom_node(
        result=bool(link and link.url),
        id=f"link_{link_index}_exists",
        desc=f"Check if link {link_index} was provided",
        parent=link_parent,
        critical=True
    )
    
    # 2. Verify the link content
    link_verification = evaluator.add_leaf(
        id=f"link_{link_index}_content",
        desc=f"Verify link {link_index} contains concrete Perplexity AI hallucination example",
        parent=link_parent,
        critical=True
    )
    
    # Updated claim focusing on hallucination
    verification_claim = (
        "This webpage contains at least one concrete, real-world example where Perplexity AI "
        "(specifically their search products like Pro Search or Deep Research) produced a hallucinated answer. "
        "The hallucination must involve fabricated URLs, made-up information, or other factually incorrect "
        "content that was presented as true by Perplexity AI."
    )
    
    # Updated instructions focusing on hallucination examples
    additional_instruction = (
        "To verify this claim, look for:\n"
        "1. HALLUCINATION EXAMPLES: The page must contain specific instances where Perplexity AI generated "
        "false information, such as:\n"
        "   - Fabricated URLs that don't exist\n"
        "   - Made-up facts, statistics, or quotes\n"
        "   - Invented sources or citations\n"
        "   - False information presented as factual\n"
        "2. CONCRETE DETAILS: The example must include:\n"
        "   - What the user asked (the query/input)\n"
        "   - What Perplexity AI responded (the hallucinated output)\n"
        "   - OR at minimum, a concrete description of both the query and the hallucinated response\n"
        "3. REAL USER EXPERIENCES: Look for actual user reports, screenshots, documented cases, "
        "or journalistic investigations with specific examples.\n"
        "4. EXCLUSIONS: General criticism, theoretical discussions about AI hallucination, "
        "or mentions of problems without specific hallucination examples do NOT count."
    )
    
    await evaluator.verify(
        claim=verification_claim,
        node=link_verification,
        sources=link.url if link else None,
        additional_instruction=additional_instruction,
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
    Evaluate a single answer for the "find_failure_perplexity" task.
    
    This function extracts URLs from the answer, verifies each URL for Perplexity AI hallucination examples,
    and determines whether the answer satisfies the task requirements.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract structured info from the answer ---------------- #
    # Extract failure links from the answer
    failure_info = await evaluator.extract(
        prompt=prompt_extract_failure_links(),
        template_class=PerplexityFailureInfo,
        extraction_name="failure_links"
    )
    
    # Process links: deduplicate and limit to first 3
    unique_links = []
    seen_urls = set()
    
    for link in failure_info.failure_links:
        if link.url and link.url not in seen_urls:
            seen_urls.add(link.url)
            unique_links.append(link)
    
    # Take only the first 3 links if more are provided
    links_to_verify = unique_links[:3] if len(unique_links) >= 3 else unique_links

    # Add custom info about link processing
    evaluator.add_custom_info({
        "total_links_extracted": len(failure_info.failure_links),
        "unique_links_count": len(unique_links),
        "links_verified_count": len(links_to_verify),
        "unique_urls": [link.url for link in unique_links],
    }, "link_statistics")

    # -------- 3. Build verification tree -------------------------------- #
    # Create a parent node for all link verifications
    links_verification_parent = evaluator.add_parallel(
        id="all_links_verification",
        desc="Verification of all provided links for Perplexity AI hallucination examples",
        parent=root,
        critical=False
    )
    
    # Verify each link (up to 3)
    for i in range(3):
        link = links_to_verify[i] if i < len(links_to_verify) else None
        await verify_link_validity(evaluator, links_verification_parent, link, i+1)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()