import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "find_three_ai_researchers"
TASK_DESCRIPTION = """Find three computer science researchers who specialize in artificial intelligence or machine learning and meet all of the following criteria:

1. Each researcher must have a public, verified Google Scholar profile
2. Each researcher must have an h-index of at least 30 (as shown on their Google Scholar profile)
3. Each researcher must be currently affiliated with a university in the United States
4. Each researcher must have an ORCID identifier

For each researcher, provide:
- Full name
- Link to their Google Scholar profile
- Their current h-index value
- Their current university affiliation
- Their ORCID identifier
- A reference URL confirming their current affiliation (e.g., university department page)
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResearcherItem(BaseModel):
    full_name: Optional[str] = None
    scholar_url: Optional[str] = None
    h_index: Optional[str] = None
    affiliation: Optional[str] = None
    affiliation_url: Optional[str] = None
    orcid: Optional[str] = None
    orcid_url: Optional[str] = None


class ResearchersExtraction(BaseModel):
    researchers: List[ResearcherItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_researchers() -> str:
    return """
Extract all researchers mentioned in the answer that are intended to satisfy the task. For each researcher, extract the following fields exactly as they appear in the answer:

- full_name: The researcher's full name
- scholar_url: The URL to their Google Scholar profile (must be a Google Scholar profile URL, e.g., https://scholar.google.com/citations?user=...)
- h_index: The h-index value stated in the answer (keep as a string, e.g., "35")
- affiliation: The current university affiliation stated in the answer (e.g., "Stanford University")
- affiliation_url: A reference URL confirming current affiliation (e.g., a university department or lab page). Extract the exact URL if present.
- orcid: The ORCID identifier as presented (e.g., "0000-0002-1825-0097" or full URL "https://orcid.org/0000-0002-1825-0097")
- orcid_url: If the answer provides a separate ORCID link, extract it here. If only an ORCID identifier is given and no URL is explicitly shown in the answer, set orcid_url to null.

RULES:
- Only extract URLs that are explicitly present in the answer text (plain URL or in markdown format). Do not invent or construct any URLs.
- If any field is missing for a researcher, set that field to null.
- Return all researchers mentioned; the evaluator will consider only the first three.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _collect_existing_urls(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if _nonempty(u)]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_researcher(
    evaluator: Evaluator,
    parent_node,
    researcher: ResearcherItem,
    idx: int,
) -> None:
    """
    Build and run the verification subtree for a single researcher.
    All concrete checks are individual leaf nodes with binary outcomes.
    """

    # Parent node for this researcher (parallel aggregation; non-critical to allow partial credit across researchers)
    rnode = evaluator.add_parallel(
        id=f"Researcher_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} researcher meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Output completeness (critical): all required fields are provided
    completeness_ok = all([
        _nonempty(researcher.full_name),
        _nonempty(researcher.scholar_url),
        _nonempty(researcher.h_index),
        _nonempty(researcher.affiliation),
        _nonempty(researcher.affiliation_url),
        _nonempty(researcher.orcid) or _nonempty(researcher.orcid_url),
    ])
    evaluator.add_custom_node(
        result=completeness_ok,
        id=f"Researcher{idx+1}_OutputCompleteness",
        desc=f"All required information fields are provided for the {['first','second','third'][idx]} researcher (full name, Google Scholar link, h-index value, university affiliation, ORCID, reference URL)",
        parent=rnode,
        critical=True
    )

    # 2) Google Scholar profile is public and verified
    gs_leaf = evaluator.add_leaf(
        id=f"Researcher{idx+1}_GoogleScholar",
        desc=f"{['First','Second','Third'][idx]} researcher has a public, verified Google Scholar profile with a provided link",
        parent=rnode,
        critical=True
    )
    gs_claim_name = researcher.full_name if _nonempty(researcher.full_name) else "the researcher"
    gs_claim = (
        f"This URL is the public Google Scholar profile of {gs_claim_name} and the profile displays a 'Verified email' badge."
    )
    await evaluator.verify(
        claim=gs_claim,
        node=gs_leaf,
        sources=researcher.scholar_url if _nonempty(researcher.scholar_url) else None,
        additional_instruction="Confirm the page is a Google Scholar profile page and look for the phrase 'Verified email' (or equivalent) on the profile."
    )

    # 3) H-index >= 30 as shown on Google Scholar
    hidx_leaf = evaluator.add_leaf(
        id=f"Researcher{idx+1}_HIndex",
        desc=f"{['First','Second','Third'][idx]} researcher has an h-index of at least 30 as shown on their Google Scholar profile",
        parent=rnode,
        critical=True
    )
    hidx_claim = "This Google Scholar profile shows an h-index of at least 30."
    await evaluator.verify(
        claim=hidx_claim,
        node=hidx_leaf,
        sources=researcher.scholar_url if _nonempty(researcher.scholar_url) else None,
        additional_instruction="Check the h-index on the profile's 'Metrics' section. If the h-index is 30 or higher, pass."
    )

    # 4) Current affiliation is a U.S. university
    us_aff_leaf = evaluator.add_leaf(
        id=f"Researcher{idx+1}_USAffiliation",
        desc=f"{['First','Second','Third'][idx]} researcher is currently affiliated with a university in the United States",
        parent=rnode,
        critical=True
    )
    aff_name = researcher.affiliation if _nonempty(researcher.affiliation) else "the stated affiliation"
    aff_claim_name = researcher.full_name if _nonempty(researcher.full_name) else "the researcher"
    us_claim = (
        f"This page indicates that {aff_claim_name} is currently affiliated with {aff_name}, and that {aff_name} is a university located in the United States."
    )
    aff_sources = _collect_existing_urls(researcher.affiliation_url, researcher.scholar_url)
    await evaluator.verify(
        claim=us_claim,
        node=us_aff_leaf,
        sources=aff_sources if aff_sources else None,
        additional_instruction=(
            "Use the affiliation page (preferred) and/or the Scholar profile to confirm current affiliation. "
            "Consider strong indicators like .edu domains, 'United States', U.S. address, or other explicit cues. "
            "If the page clearly shows a U.S. university affiliation, pass."
        )
    )

    # 5) ORCID identifier is valid (prefer page-based verification if a URL is provided)
    orcid_leaf = evaluator.add_leaf(
        id=f"Researcher{idx+1}_ORCID",
        desc=f"{['First','Second','Third'][idx]} researcher has a valid ORCID identifier provided",
        parent=rnode,
        critical=True
    )
    if _nonempty(researcher.orcid_url):
        orcid_claim_name = researcher.full_name if _nonempty(researcher.full_name) else "the researcher"
        orcid_claim = f"This page is the ORCID record for {orcid_claim_name}."
        await evaluator.verify(
            claim=orcid_claim,
            node=orcid_leaf,
            sources=researcher.orcid_url,
            additional_instruction="Verify that the page is on orcid.org and that the displayed name matches (allow minor spelling or formatting variations)."
        )
    else:
        # Fall back to format validation if only an identifier is provided without URL
        orcid_id_text = researcher.orcid if _nonempty(researcher.orcid) else ""
        orcid_claim = (
            f"The identifier '{orcid_id_text}' is a valid ORCID iD format (16 digits grouped by hyphens, e.g., 0000-0002-1825-0097)."
        )
        await evaluator.verify(
            claim=orcid_claim,
            node=orcid_leaf,
            sources=None,
            additional_instruction="Judge only the format validity of the ORCID iD. Accept 16-digit identifiers with hyphens in groups of four (check digit may be 'X')."
        )

    # 6) Specialization in AI or ML
    aiml_leaf = evaluator.add_leaf(
        id=f"Researcher{idx+1}_AIMLSpecialization",
        desc=f"{['First','Second','Third'][idx]} researcher specializes in artificial intelligence or machine learning",
        parent=rnode,
        critical=True
    )
    spec_claim_name = researcher.full_name if _nonempty(researcher.full_name) else "the researcher"
    aiml_claim = (
        f"This page indicates that {spec_claim_name} specializes in artificial intelligence or machine learning."
    )
    aiml_sources = _collect_existing_urls(researcher.scholar_url, researcher.affiliation_url)
    await evaluator.verify(
        claim=aiml_claim,
        node=aiml_leaf,
        sources=aiml_sources if aiml_sources else None,
        additional_instruction=(
            "Look for research interests or descriptions explicitly mentioning 'artificial intelligence' or 'machine learning'. "
            "Also accept well-recognized AI/ML subfields (e.g., deep learning, computer vision, natural language processing, reinforcement learning) as evidence."
        )
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
    Evaluate an answer for the 'Find three AI/ML researchers' task.
    """
    # Initialize evaluator with a parallel root to allow partial credit across researchers
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

    # Extract researchers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_researchers(),
        template_class=ResearchersExtraction,
        extraction_name="researchers_extraction"
    )

    # Only evaluate the first three researchers; pad with placeholders if fewer than three
    items: List[ResearcherItem] = list(extracted.researchers[:3])
    while len(items) < 3:
        items.append(ResearcherItem())

    # Build and evaluate subtree for each researcher
    for i in range(3):
        await verify_researcher(
            evaluator=evaluator,
            parent_node=root,
            researcher=items[i],
            idx=i
        )

    # Return the evaluation summary
    return evaluator.get_summary()