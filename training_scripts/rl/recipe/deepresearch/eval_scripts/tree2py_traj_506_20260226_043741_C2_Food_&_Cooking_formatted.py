import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thanksgiving_breakfast_chains_2024"
TASK_DESCRIPTION = """
Identify two national breakfast restaurant chains that were confirmed to be open on Thanksgiving Day 2024 with consistent nationwide operating hours starting at or before 8:00 AM. The chains must serve breakfast food and must have had a verifiable, consistent nationwide policy (not location-dependent hours) for Thanksgiving 2024.
"""

THANKSGIVING_2024_DATE = "Thursday, November 28, 2024"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChainCandidate(BaseModel):
    name: Optional[str] = None
    thanksgiving_open_time: Optional[str] = None  # e.g., "6:00 AM", "24/7", "7 a.m."
    nationwide_policy_summary: Optional[str] = None  # short summary the answer claims
    sources: List[str] = Field(default_factory=list)  # all URLs cited for this chain


class ChainsExtraction(BaseModel):
    chains: List[ChainCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chains() -> str:
    return """
    Extract up to the first two restaurant chains that the answer claims meet the Thanksgiving 2024 criteria.
    For each identified chain, extract the following fields:
    - name: The chain’s name.
    - thanksgiving_open_time: The opening time the answer claims for Thanksgiving Day 2024 (e.g., "6:00 AM", "7am", "24/7"). If not explicitly stated, return null.
    - nationwide_policy_summary: A short paraphrase of the claimed nationwide Thanksgiving 2024 hours/policy (e.g., "all locations open 24/7", "company announced all stores open at 7am"). If not stated, return null.
    - sources: Extract all URLs (including markdown links) the answer cites for this chain that are intended to support Thanksgiving 2024 operations/hours. If none are provided, return an empty list.
    
    Return a JSON object:
    {
      "chains": [
        { "name": ..., "thanksgiving_open_time": ..., "nationwide_policy_summary": ..., "sources": [...] },
        { "name": ..., "thanksgiving_open_time": ..., "nationwide_policy_summary": ..., "sources": [...] }
      ]
    }
    
    Rules:
    - Only extract information that is explicitly present in the answer.
    - For URLs, extract the actual link; do not invent any.
    - If the answer lists more than two candidate chains, extract only the first two mentioned.
    - If the answer has fewer than two, extract whatever is available and omit the rest.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_chain(
    evaluator: Evaluator,
    parent_node,
    chain: ChainCandidate,
    index: int,
) -> None:
    """
    Build the verification subtree for a single chain according to the rubric.
    Structure:
    - First_Chain / Second_Chain (sequential, NON-CRITICAL)
      - Chain_Qualification (parallel, CRITICAL)
        - National_Breakfast_Chain (leaf, CRITICAL)
        - Verifiable_Source (leaf, CRITICAL)
      - Thanksgiving_2024_Hours (sequential, CRITICAL)
        - Opens_Early (leaf, CRITICAL)
    """
    pretty_idx = "First" if index == 0 else "Second"
    chain_label = chain.name or "the identified chain"

    # Container for this chain (sequential, non-critical)
    chain_node = evaluator.add_sequential(
        id=f"{pretty_idx.lower()}_chain",
        desc=f"{pretty_idx} restaurant chain meeting all Thanksgiving 2024 criteria",
        parent=parent_node,
        critical=False,
    )

    # 1) Chain Qualification (parallel, critical)
    qual_node = evaluator.add_parallel(
        id=f"{pretty_idx.lower()}_qualification",
        desc="Chain qualifies as a national breakfast restaurant with verifiable information",
        parent=chain_node,
        critical=True,
    )

    # 1.a) National breakfast chain (leaf, critical)
    national_leaf = evaluator.add_leaf(
        id=f"{pretty_idx.lower()}_national_breakfast_chain",
        desc="Is a national restaurant chain (operates in multiple U.S. states) that serves breakfast food",
        parent=qual_node,
        critical=True,
    )

    national_claim = (
        f"{chain_label} is a national restaurant chain that operates in multiple U.S. states and serves breakfast food."
    )
    await evaluator.verify(
        claim=national_claim,
        node=national_leaf,
        sources=chain.sources,
        additional_instruction=(
            "Use only the provided sources. Consider official company pages (about, menu, locations) or credible articles. "
            "National means presence in multiple U.S. states (not just a local chain). "
            "Serving breakfast should be a core offering (breakfast concept, breakfast menu, or 24/7 diner serving breakfast)."
        ),
    )

    # 1.b) Verifiable source (leaf, critical)
    verifiable_leaf = evaluator.add_leaf(
        id=f"{pretty_idx.lower()}_verifiable_source",
        desc="Information is verifiable through official company sources or credible 2024 news reports",
        parent=qual_node,
        critical=True,
    )

    verifiable_claim = (
        f"This URL is either an official company source for {chain_label} "
        f"(company website, newsroom/press release, or official social account) "
        f"or a credible news article published in 2024 that reports Thanksgiving 2024 opening status/hours for {chain_label}."
    )
    await evaluator.verify(
        claim=verifiable_claim,
        node=verifiable_leaf,
        sources=chain.sources,
        additional_instruction=(
            "Pass if at least one of the provided URLs satisfies this. "
            "Prefer official corporate domains or well-known news outlets. "
            "The page should clearly relate to Thanksgiving 2024. "
            "If no URLs are provided, this should not be supported."
        ),
    )

    # 2) Thanksgiving 2024 Hours (sequential, critical)
    hours_node = evaluator.add_sequential(
        id=f"{pretty_idx.lower()}_thanksgiving_2024_hours",
        desc="Restaurant's Thanksgiving 2024 operating hours meet requirements",
        parent=chain_node,
        critical=True,
    )

    # 2.a) Opens early with consistent nationwide policy (leaf, critical)
    opens_leaf = evaluator.add_leaf(
        id=f"{pretty_idx.lower()}_opens_early",
        desc="Chain opened at or before 8:00 AM on Thanksgiving Day 2024 with a consistent nationwide policy",
        parent=hours_node,
        critical=True,
    )

    opens_time_str = chain.thanksgiving_open_time or "8:00 AM or earlier"
    opens_claim = (
        f"On {THANKSGIVING_2024_DATE}, {chain_label} had a consistent nationwide policy to be open and to open at or "
        f"before 8:00 AM across U.S. locations (for example, standard opening time {opens_time_str} or 24/7 operations). "
        f"This was a chain-wide policy, not location-dependent hours."
    )
    await evaluator.verify(
        claim=opens_claim,
        node=opens_leaf,
        sources=chain.sources,
        additional_instruction=(
            "Support the following simultaneously from the provided URLs: "
            "1) The chain was open on Thanksgiving Day 2024; "
            "2) Opening time was 8:00 AM or earlier (treat 24/7 as qualifying); "
            "3) The policy was consistent chain‑wide/nationwide, not merely 'hours vary by location'. "
            "Reject if the evidence only shows individual stores or general 'may vary by location' without a consistent nationwide policy."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Thanksgiving 2024 breakfast chains task.
    """
    # Initialize evaluator with a parallel root (two chains evaluated independently)
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

    # Extract up to two chain candidates from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=ChainsExtraction,
        extraction_name="chains_extraction",
    )

    # Normalize to exactly two entries (pad with empty if needed; trim if more)
    chains: List[ChainCandidate] = list(extracted.chains[:2])
    while len(chains) < 2:
        chains.append(ChainCandidate())

    # Build the verification tree according to the rubric
    # Top-level node mirroring the rubric's main container (optional but clearer)
    top = evaluator.add_parallel(
        id="two_chains_main",
        desc="Identify two national breakfast restaurant chains that were open on Thanksgiving Day 2024 with specific hour requirements",
        parent=root,
        critical=False,
    )

    # First chain subtree
    await verify_chain(evaluator, top, chains[0], index=0)

    # Second chain subtree
    await verify_chain(evaluator, top, chains[1], index=1)

    # Return evaluation summary
    return evaluator.get_summary()