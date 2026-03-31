import asyncio
import logging
from typing import Optional, List, Dict, Union

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chris_evans"
TASK_DESCRIPTION = """
Determine if actor Chris Evans can be connected to any U.S. president born prior to the year 1920 through direct political relationships or familial relationships within three degrees of separation. If such a connection exists, please specify:

1. The name and birth date of the U.S. president.
2. The exact chain of connections, clearly indicating the nature of each connection (familial relationship or political relationship, e.g., relative, colleague, successor, mentor).
"""

JUDGE_MODEL = "o4-mini"

# Ground truth information
GROUND_TRUTH_CHAINS = [
    "Chris Evans is the nephew of Mike Capuano",
    ["Mike Capuano was preceded by Joseph P. Kennedy II as a member of US House Representatives", "Mike Capuano is/was colleague of Edward Moore Kennedy (Ted Kennedy)"],
    ["Joseph P. Kennedy II is a nephew of US President John F. Kennedy", "Edward Moore Kennedy (Ted Kennedy) is a brother of US President John F. Kennedy"]
]
GROUND_TRUTH_PRESIDENT = "John F. Kennedy"
GROUND_TRUTH_BIRTH_YEAR = 1917


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BasicConnectionResult(BaseModel):
    """Basic result of whether a connection was found"""
    connection_found: bool
    summary: Optional[str]


class ConnectionLink(BaseModel):
    """Represents one link in the connection chain"""
    person1: str
    person2: str
    relationship_description: str


class ConnectionChain(BaseModel):
    """Represents the full connection chain"""
    links: List[ConnectionLink] = Field(default_factory=list)


class PresidentInfo(BaseModel):
    """Information about the connected president"""
    name: Optional[str]
    birth_date: Optional[str]
    birth_year: Optional[int]


class SourceLinks(BaseModel):
    """URLs provided as sources in the answer"""
    links: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_basic_result() -> str:
    return """
    First, determine if the answer found any connection between Chris Evans and a U.S. president born before 1920.

    Extract:
    1. Whether a connection was found (true/false)
    2. If found, provide a brief summary of what connection was claimed

    Only extract what is explicitly stated in the answer.
    """


def prompt_extract_connection_chain() -> str:
    return """
    Extract the detailed connection chain described in the answer.

    For each step in the connection chain, extract:
    - The two people being connected in this step
    - The full description of their relationship as stated in the answer, as detailed as possible as how it is phrased in the answer, including but not limited to: the person names, the relationship types, the full descriptions and all backgrounds or context. This field is not limited to one sentence, but can be multiple sentences if there are multiple sentences in the answer

    List the connections in the order they appear in the chain from Chris Evans to the president.
    Only extract connections that are explicitly described.
    """


def prompt_extract_president_info() -> str:
    return """
    Extract information about the U.S. president mentioned in the connection:
    - President's full name
    - Birth date (if provided, including full date or just year)
    - Birth year as a number (if mentioned)

    Extract only what is explicitly stated in the answer.
    """


def prompt_extract_sources_for_claim(claim: str) -> str:
    return f"""
    Extract all URLs from the answer that are cited as sources or could potentially support this specific claim:

    "{claim}"

    Only extract URLs that are explicitly provided in the answer as sources or references.
    Return them as a simple list of URL strings.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_connection_link(
        evaluator: Evaluator,
        parent_node,
        link: ConnectionLink,
        link_index: int,
        ground_truth_connection: Union[str, List],
) -> None:
    """
    Verify a single connection link with both correctness and provenance.
    """
    # Create parent node for this connection
    connection_node = evaluator.add_parallel(
        id=f"connection_link_{link_index}",
        desc=f"Connection step {link_index}: {link.person1} to {link.person2} relationship is correct and supported",
        parent=parent_node,
        critical=False,
    )

    # Create claim description
    claim_text = f"{link.person1} has a relationship with {link.person2}: {link.relationship_description}"

    # Extract sources for this specific claim first
    sources = await evaluator.extract(
        prompt=prompt_extract_sources_for_claim(claim_text),
        template_class=SourceLinks,
        extraction_name=f"sources_link_{link_index}"
    )

    # 1. Combined existence check for link info and sources
    link_exists_node = evaluator.add_custom_node(
        result=(link.person1 and link.person2 and link.relationship_description) and bool(sources.links),
        id=f"link_{link_index}_info_exists",
        desc=f"Check if connection step {link_index} has both relationship description and supporting URLs",
        parent=connection_node,
        critical=True
    )

    # 2. Verify correctness against ground truth
    correctness_node = evaluator.add_leaf(
        id=f"link_{link_index}_correctness",
        desc=f"Connection step {link_index} matches the expected relationship",
        parent=connection_node,
        critical=True
    )

    # Check against specific ground truth connection
    if isinstance(ground_truth_connection, str):
        await evaluator.verify(
            claim=f"Here is a description of a relationship:  ({claim_text}). Check whether the relationship described in this piece of text matches or covers or includes this relationship: {ground_truth_connection}",
            node=correctness_node,
            additional_instruction="Check if the relationship described matches or covers the expected connection. Allow for reasonable variations in wording and phrasing as long as the core relationship (people and connection type or connection essence) is essentially the same. Don't be too strict on the exact wording or exact semantic matching for all the details"
        )
    else:
        # For multiple possible ground truth connections, try each one
        ground_truth_match = False
        for single_gt_connection in ground_truth_connection:
            match_result = await evaluator.verify(
                claim=f"Here is a description of a relationship: ({claim_text}). Check whether the relationship described in this piece of text matches or covers or includes this relationship: {single_gt_connection}",
                node=None,  # Don't assign to node yet, just get result
                additional_instruction="Check if the relationship described matches or covers the expected connection. Allow for reasonable variations in wording and phrasing as long as the core relationship (people and connection type or connection essence) is essentially the same. Don't be too strict on the exact wording or exact semantic matching for all the details"
            )
            if match_result:
                ground_truth_match = True
                break
        
        # Manually assign the result to the node
        correctness_node.score = 1.0 if ground_truth_match else 0.0
        correctness_node.status = "passed" if ground_truth_match else "failed"

    # 3. Verify provenance (source support)
    provenance_node = evaluator.add_leaf(
        id=f"link_{link_index}_provenance",
        desc=f"Connection step {link_index} is supported by cited sources",
        parent=connection_node,
        critical=True
    )

    await evaluator.verify(
        claim=claim_text,
        node=provenance_node,
        sources=sources.links,
        additional_instruction="really check into details of the provided webpage. Find any evidence that may support the fact. Don't be too strict on the exact wording or exact semantic matching for all the details"
    )


async def verify_connection_chain_sequential(
        evaluator: Evaluator,
        parent_node,
        connection_chain: ConnectionChain,
) -> None:
    """
    Verify the three-step connection chain sequentially.
    Each step must pass before the next one is evaluated.
    """
    # Get the provided links (up to 3)
    provided_links = connection_chain.links[:3] if connection_chain.links else []

    # Verify each of the 3 steps
    for i in range(3):
        if i < len(provided_links):
            # Verify the provided link
            await verify_connection_link(
                evaluator=evaluator,
                parent_node=parent_node,
                link=provided_links[i],
                link_index=i + 1,
                ground_truth_connection=GROUND_TRUTH_CHAINS[i],
            )
        else:
            # Create node for missing connection
            missing_node = evaluator.add_leaf(
                id=f"connection_link_{i + 1}",
                desc=f"Connection step {i + 1} is provided and correct",
                parent=parent_node,
                critical=False,
                score=0.0,
                status="skipped"
            )


async def verify_president_info(
        evaluator: Evaluator,
        parent_node,
        president_info: PresidentInfo,
) -> None:
    """
    Verify the president information is correct and supported by sources.
    """
    # Create parent node for president verification
    president_node = evaluator.add_parallel(
        id="president_verification",
        desc="Information about the connected U.S. president is correct and supported",
        parent=parent_node,
        critical=False,
    )

    # Extract sources for president information first
    president_claim = f"{president_info.name} was a U.S. president born in {president_info.birth_year or 1917}"
    sources = await evaluator.extract(
        prompt=prompt_extract_sources_for_claim(president_claim),
        template_class=SourceLinks,
        extraction_name="president_sources"
    )

    # 1. Combined existence check for president info and sources
    president_exists_node = evaluator.add_custom_node(
        result=(president_info.name is not None and president_info.name.strip() != "") and bool(sources.links),
        id="president_info_exists",
        desc="Check if both president name and supporting URLs were provided in the answer",
        parent=president_node,
        critical=True
    )

    # 2. Verify president identity
    identity_node = evaluator.add_leaf(
        id="president_identity",
        desc="The identified president is John F. Kennedy",
        parent=president_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The president identified ({president_info.name}) is John F. Kennedy",
        node=identity_node,
        additional_instruction="Check if the name corresponds to John F. Kennedy, allowing for common variations like 'JFK', 'John Kennedy', 'John Fitzgerald Kennedy', etc."
    )

    # 3. Create parent node for birth information verification
    birth_parent_node = evaluator.add_parallel(
        id="birth_verification",
        desc="President birth information verification",
        parent=president_node,
        critical=False,
    )

    # Birth info existence check
    birth_exists_node = evaluator.add_custom_node(
        result=(president_info.birth_year is not None) or (president_info.birth_date is not None and president_info.birth_date.strip() != ""),
        id="birth_info_exists",
        desc="Check if birth information (year or date) was provided",
        parent=birth_parent_node,
        critical=True
    )

    # Verify birth information
    birth_node = evaluator.add_leaf(
        id="president_birth_info",
        desc="The president's birth information is accurate",
        parent=birth_parent_node,
        critical=True
    )

    # Choose which claim to verify based on available info
    if president_info.birth_year:
        await evaluator.verify(
            claim=f"John F. Kennedy was born in {president_info.birth_year}",
            node=birth_node,
            additional_instruction="The correct birth year is 1917. Check if the provided year matches."
        )
    elif president_info.birth_date and president_info.birth_date.strip():
        await evaluator.verify(
            claim=f"John F. Kennedy's birth date is {president_info.birth_date}",
            node=birth_node,
            additional_instruction="The correct birth date is May 29, 1917. Check if the provided date matches."
        )

    # 4. Verify source support for president info
    source_node = evaluator.add_leaf(
        id="president_source_support",
        desc="President information is supported by cited sources",
        parent=president_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"{president_info.name} was a U.S. president and was born in 1917, which aligns with the information in this webpage",
        node=source_node,
        sources=sources.links,
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

    # -------- 2. Extract information step by step ----------------------- #
    # Step 1: Check if connection was found
    basic_result = await evaluator.extract(
        prompt=prompt_extract_basic_result(),
        template_class=BasicConnectionResult,
        extraction_name="basic_result"
    )

    # Step 2: Extract connection chain (if connection found)
    connection_chain = ConnectionChain(links=[])
    if basic_result.connection_found:
        connection_chain = await evaluator.extract(
            prompt=prompt_extract_connection_chain(),
            template_class=ConnectionChain,
            extraction_name="connection_chain"
        )

    # Step 3: Extract president information (if connection found)
    president_info = PresidentInfo(name=None, birth_date=None, birth_year=None)
    if basic_result.connection_found:
        president_info = await evaluator.extract(
            prompt=prompt_extract_president_info(),
            template_class=PresidentInfo,
            extraction_name="president_info"
        )

    # -------- 3. Build sequential verification tree --------------------- #
    # Sequential step 1: Connection chain verification (3 steps)
    connection_chain_node = evaluator.add_sequential(
        id="connection_chain",
        desc="The three-step connection chain from Chris Evans to the president is correct and supported",
        critical=False,
    )

    # Verify the connection chain sequentially
    await verify_connection_chain_sequential(
        evaluator=evaluator,
        parent_node=connection_chain_node,
        connection_chain=connection_chain,
    )

    # Sequential step 2: President information verification
    await verify_president_info(
        evaluator=evaluator,
        parent_node=root,
        president_info=president_info,
    )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()