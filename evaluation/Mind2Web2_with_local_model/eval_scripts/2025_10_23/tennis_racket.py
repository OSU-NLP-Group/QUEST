import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tennis_racket"
TASK_DESCRIPTION = """
I would like to purchase a tennis racket online. Please provide a list of three currently available tennis rackets that are either endorsed or used by professional players on the ATP or WTA tours. For each racket, include a source that confirms its use or endorsement by a professional player. Additionally, provide at least two purchase links for each racket from different online retailers.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                      #
# --------------------------------------------------------------------------- #
class RacketBasic(BaseModel):
    """Basic information about a tennis racket."""
    name: Optional[str] = None
    player_name: Optional[str] = None


class RacketsList(BaseModel):
    """List of basic tennis rackets extracted from the answer."""
    rackets: List[RacketBasic] = Field(default_factory=list)


class EndorsementLinks(BaseModel):
    """Source URLs confirming endorsement."""
    endorsement_urls: List[str] = Field(default_factory=list)


class PurchaseLink(BaseModel):
    """Purchase link for a tennis racket."""
    retailer: Optional[str] = None
    url: Optional[str] = None


class PurchaseLinks(BaseModel):
    """List of purchase links for a tennis racket."""
    purchase_links: List[PurchaseLink] = Field(default_factory=list)


class RetailerInfo(BaseModel):
    """Information about a retailer extracted from a URL."""
    retailer_name: Optional[str] = None
    website_domain: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_rackets_list() -> str:
    return """
    Extract information about the tennis rackets mentioned in the answer. For each racket, include:
    1. The name of the racket (including brand and model)
    2. The name of the professional player who endorses or uses the racket

    Return this information as a structured JSON object with a list of rackets.

    If any information is missing, set the corresponding field to null.
    """


def prompt_extract_endorsement_links(racket_name: str, player_name: str) -> str:
    return f"""
    Extract all URLs provided in the answer that confirm or show that the professional player "{player_name}" endorses or uses the tennis racket "{racket_name}".

    Return a list of all URLs mentioned that serve as evidence for this endorsement.
    If no URLs are provided, return an empty list.
    """


def prompt_extract_purchase_links(racket_name: str) -> str:
    return f"""
    Extract all purchase links provided in the answer for the tennis racket "{racket_name}".

    For each purchase link, extract:
    1. The name of the retailer or website (if mentioned)
    2. The URL to purchase the racket

    Return this information as a structured JSON object with a list of purchase links.
    If no purchase links are provided, return an empty list.
    """


def prompt_extract_retailer_info(url: str) -> str:
    return f"""
    Extract information about the retailer from this URL: {url}

    Please identify:
    1. The retailer_name: The name of the company or retailer that operates this website
       (For example, if it's Amazon, return "Amazon". If it's Tennis Warehouse, return "Tennis Warehouse")
    2. The website_domain: The main domain of the website (e.g., "amazon.com", "tenniswarehouse.com")

    Return this information as a structured JSON object.
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def verify_endorsement(
        evaluator: Evaluator,
        parent_node,
        racket_name: str,
        player_name: str,
        endorsement_links: List[str],
        index: int
) -> None:
    """
    Verify that the tennis racket is endorsed or used by a professional player with valid source URLs.
    """
    # Create parent node for endorsement verification
    endorsement_parent = evaluator.add_parallel(
        id=f"endorsement_{index}",
        desc=f"Endorsement verification for racket {index}: {racket_name}",
        parent=parent_node,
        critical=True,
    )

    # Critical existence check
    exists_node = evaluator.add_custom_node(
        result=bool(racket_name and player_name and endorsement_links),
        id=f"endorsement_exists_{index}",
        desc=f"Check if endorsement information exists for racket {index}",
        parent=endorsement_parent,
        critical=True
    )

    # Actual endorsement verification
    verification_node = evaluator.add_leaf(
        id=f"endorsement_verification_{index}",
        desc=f"Verify {player_name} endorses/uses {racket_name}",
        parent=endorsement_parent,
        critical=True
    )

    claim = f"The tennis racket {racket_name} is endorsed or used by the professional tennis player {player_name}."
    await evaluator.verify(
        claim=claim,
        node=verification_node,
        sources=endorsement_links,
        additional_instruction="Verify that the provided URLs confirm the endorsement or use of this racket by the professional player."
    )


async def verify_purchase_link(
        evaluator: Evaluator,
        parent_node,
        racket_name: str,
        url: str,
        link_index: int
) -> None:
    """
    Verify that a purchase link is valid for the tennis racket.
    """
    # Create parent node for this purchase link
    link_parent = evaluator.add_parallel(
        id=f"purchase_link_{link_index}",
        desc=f"Purchase link {link_index} verification",
        parent=parent_node,
        critical=False,
    )

    # Critical existence check
    exists_node = evaluator.add_custom_node(
        result=bool(racket_name and url),
        id=f"purchase_link_exists_{link_index}",
        desc=f"Check if purchase link {link_index} exists",
        parent=link_parent,
        critical=True
    )

    # Actual purchase link verification
    verification_node = evaluator.add_leaf(
        id=f"purchase_link_verification_{link_index}",
        desc=f"Verify purchase link {link_index} for {racket_name}",
        parent=link_parent,
        critical=True
    )

    claim = f"This URL is a valid purchase link for the tennis racket {racket_name}."
    await evaluator.verify(
        claim=claim,
        node=verification_node,
        sources=url,
        additional_instruction="Verify that this URL leads to a page where the specified tennis racket can be purchased."
    )


async def verify_different_retailers(
        evaluator: Evaluator,
        parent_node,
        racket_name: str,
        purchase_links: List[PurchaseLink],
        index: int
) -> None:
    """
    Verify that the purchase links are from different retailers.
    """
    # Create parent node for different retailers verification
    retailers_parent = evaluator.add_parallel(
        id=f"different_retailers_{index}",
        desc=f"Different retailers verification for racket {index}",
        parent=parent_node,
        critical=True,
    )

    # Critical existence check
    exists_node = evaluator.add_custom_node(
        result=(len(purchase_links) >= 2 and 
                bool(purchase_links[0].url) and 
                bool(purchase_links[1].url)),
        id=f"retailers_check_exists_{index}",
        desc=f"Check if two purchase links exist for retailer comparison",
        parent=retailers_parent,
        critical=True
    )

    # Extract retailer information from the first two purchase links
    retailer_info1 = await evaluator.extract(
        prompt=prompt_extract_retailer_info(purchase_links[0].url),
        template_class=RetailerInfo,
        extraction_name=f"retailer_info_1_{index}",
        source=purchase_links[0].url
    )

    retailer_info2 = await evaluator.extract(
        prompt=prompt_extract_retailer_info(purchase_links[1].url),
        template_class=RetailerInfo,
        extraction_name=f"retailer_info_2_{index}",
        source=purchase_links[1].url
    )

    # Verify they are different retailers
    verification_node = evaluator.add_leaf(
        id=f"different_retailers_verification_{index}",
        desc=f"Verify purchase links are from different retailers",
        parent=retailers_parent,
        critical=True
    )

    retailer1_name = retailer_info1.retailer_name or "unknown retailer"
    retailer1_domain = retailer_info1.website_domain or "unknown domain"
    retailer2_name = retailer_info2.retailer_name or "unknown retailer"
    retailer2_domain = retailer_info2.website_domain or "unknown domain"

    claim = f"""
    The first purchase link for {racket_name} is from {retailer1_name} (domain: {retailer1_domain}).
    The second purchase link for {racket_name} is from {retailer2_name} (domain: {retailer2_domain}).
    These are different retailers, not the same company or store operating under different names.
    """

    await evaluator.verify(
        claim=claim,
        node=verification_node,
        additional_instruction="Determine if these are genuinely different retailers or online stores, not just different domains or websites owned by the same company."
    )


async def verify_racket(
        evaluator: Evaluator,
        parent_node,
        racket: RacketBasic,
        endorsement_links: EndorsementLinks,
        purchase_links_data: PurchaseLinks,
        index: int
) -> None:
    """
    Verify a single racket with all its requirements.
    """
    racket_name = racket.name or f"Unnamed Racket {index}"
    player_name = racket.player_name or "Unknown Player"

    # Create node for this racket
    racket_node = evaluator.add_parallel(
        id=f"racket_{index}",
        desc=f"Verification of tennis racket {index}: {racket_name}",
        parent=parent_node,
        critical=False,
    )

    # 1. Endorsement verification
    await verify_endorsement(
        evaluator,
        racket_node,
        racket_name,
        player_name,
        endorsement_links.endorsement_urls,
        index
    )

    # 2. Purchase links verification
    purchase_links_node = evaluator.add_parallel(
        id=f"purchase_links_{index}",
        desc=f"Purchase links verification for racket {index}",
        parent=racket_node,
        critical=False,
    )

    # Ensure we have exactly 2 purchase links (pad if needed)
    purchase_links = purchase_links_data.purchase_links[:2]
    while len(purchase_links) < 2:
        purchase_links.append(PurchaseLink())

    # Verify each purchase link
    for j, link in enumerate(purchase_links):
        await verify_purchase_link(
            evaluator,
            purchase_links_node,
            racket_name,
            link.url,
            j + 1
        )

    # 3. Different retailers verification
    await verify_different_retailers(
        evaluator,
        racket_node,
        racket_name,
        purchase_links,
        index
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
    Evaluate a single answer to the tennis racket task and return a structured result dictionary.

    This evaluation checks for:
    1. Three tennis rackets endorsed/used by professional players with valid sources
    2. Each racket has at least two purchase links from different retailers
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

    # -------- 2. Extract all information --------------------------------- #
    # Extract the basic list of rackets
    rackets_list = await evaluator.extract(
        prompt=prompt_extract_rackets_list(),
        template_class=RacketsList,
        extraction_name="rackets_list"
    )

    # Ensure we have exactly 3 rackets (pad if needed)
    rackets_to_evaluate = rackets_list.rackets[:3]
    while len(rackets_to_evaluate) < 3:
        rackets_to_evaluate.append(RacketBasic())

    # -------- 3. Extract and verify each racket ------------------------- #
    for i, racket in enumerate(rackets_to_evaluate):
        # Extract endorsement links for this racket
        endorsement_links_data = await evaluator.extract(
            prompt=prompt_extract_endorsement_links(
                racket.name or f"Racket {i + 1}",
                racket.player_name or "Unknown Player"
            ),
            template_class=EndorsementLinks,
            extraction_name=f"endorsement_links_{i + 1}"
        )

        # Extract purchase links for this racket
        purchase_links_data = await evaluator.extract(
            prompt=prompt_extract_purchase_links(racket.name or f"Racket {i + 1}"),
            template_class=PurchaseLinks,
            extraction_name=f"purchase_links_{i + 1}"
        )

        # Verify this racket
        await verify_racket(
            evaluator,
            root,
            racket,
            endorsement_links_data,
            purchase_links_data,
            i + 1
        )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()