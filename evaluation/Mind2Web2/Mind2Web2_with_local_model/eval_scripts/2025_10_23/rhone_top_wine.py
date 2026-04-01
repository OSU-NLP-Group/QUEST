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
TASK_ID = "rhone_top_wine"
TASK_DESCRIPTION = """
Could you list five wines from the Rhône region (including both Southern and Northern Rhône) that have appeared in Wine Spectator's Top 10 Wines since 2019? For each of them, please include its name, region, year it was awarded, its ranking on that year's list and a direct link to its entry on the corresponding year's Wine Spectator Top 10 list.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WineNamesList(BaseModel):
    """List of wine names identified in the answer"""
    wine_names: List[str] = Field(default_factory=list)


class WineDetails(BaseModel):
    """Details for a specific wine"""
    region: Optional[str] = None
    award_year: Optional[str] = None
    ranking: Optional[str] = None
    link: Optional[str] = None


class WineSourceUrls(BaseModel):
    """URLs related to a specific wine"""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_wine_names() -> str:
    return """
    Extract a list of wine names from the answer. Focus only on wines that are presented as Rhône wines that 
    appeared in Wine Spectator's Top 10 Wines since 2019.

    Extract the complete wine name including producer/winery name and wine name (e.g., "Domaine Jean-Louis Chave Hermitage").
    Do not include vintage years, rankings, or other details - only the wine name itself.

    If no qualifying wines are mentioned, return an empty list.
    """


def prompt_extract_wine_details(wine_name: str) -> str:
    return f"""
    Extract the details for the wine named "{wine_name}" from the answer. Look for the following information:

    - region: The specific region within Rhône (e.g., "Northern Rhône", "Southern Rhône", or specific appellations)
    - award_year: The year it was awarded or appeared in Wine Spectator's Top 10 list
    - ranking: The specific ranking position (e.g., "#1", "2", "No. 3") in the Wine Spectator's Top 10 list
    - link: The URL link to its entry on Wine Spectator's website if provided

    If any of these details are not mentioned for this specific wine, return null for that field.
    """


def prompt_extract_wine_urls(wine_name: str) -> str:
    return f"""
    Extract all URLs that are related to the wine named "{wine_name}" from the answer.

    Focus on URLs that might provide information about:
    - This specific wine's inclusion in Wine Spectator's Top 10 list
    - Information about this wine's origin in the Rhône region
    - Wine Spectator reviews or ratings for this wine
    - Any other source that provides information about this specific wine

    Return an empty list if no relevant URLs are found for this wine.
    """


# --------------------------------------------------------------------------- #
# Wine verification functions                                                 #
# --------------------------------------------------------------------------- #
async def verify_single_wine(
        evaluator: Evaluator,
        parent_node,
        wine_name: str,
        wine_details: WineDetails,
        wine_urls: List[str],
        wine_index: int,
) -> None:
    """
    Verify all aspects of a single wine entry.
    """
    wine_node = evaluator.add_parallel(
        id=f"wine_{wine_index}",
        desc=f"Verification for wine {wine_index + 1}: {wine_name if wine_name else 'Missing'}",
        parent=parent_node,
    )

    # Single completeness check for the entire wine entry
    completeness_check = evaluator.add_custom_node(
        result=bool(wine_name and wine_name.strip()),
        id=f"wine_{wine_index}_exists",
        desc=f"Check if wine {wine_index + 1} information exists",
        parent=wine_node,
        critical=True
    )

    # Origin verification - region MUST be provided per task requirements
    origin_info_exists = evaluator.add_custom_node(
        result=bool(wine_details.region) and bool(wine_urls),
        id=f"wine_{wine_index}_origin_info_exists",
        desc="Check if region information and source URLs are provided",
        parent=wine_node,
        critical=True
    )

    origin_node = evaluator.add_leaf(
        id=f"wine_{wine_index}_origin",
        desc=f"Verify that the provided region is part of Rhône",
        parent=wine_node,
        critical=True,
    )

    claim = f"The region '{wine_details.region}' is part of the Rhône wine region in France (including both Southern and Northern Rhône)"
    await evaluator.verify(
        claim=claim,
        node=origin_node,
        additional_instruction="Allow for variations in region names, such as 'Northern Rhône', 'Southern Rhône', or specific appellations like 'Côte-Rôtie' or 'Châteauneuf-du-Pape'.",
    )

    # URL provenance verification for origin
    origin_url_node = evaluator.add_leaf(
        id=f"wine_{wine_index}_origin_url_verify",
        desc=f"Verify wine origin from source URLs",
        parent=wine_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The wine {wine_name} is confirmed to be from the region '{wine_details.region}' in the Rhône wine region",
        node=origin_url_node,
        sources=wine_urls,
        additional_instruction="Verify that the source confirms this wine is from the stated Rhône region.",
    )

    # Top 10 verification - year, ranking, and URLs MUST be provided
    top10_info_exists = evaluator.add_custom_node(
        result=bool(wine_details.ranking and wine_details.award_year and wine_urls),
        id=f"wine_{wine_index}_top10_info_exists",
        desc="Check if ranking, year, and source URLs are provided",
        parent=wine_node,
        critical=True
    )

    # Verify year is valid (2019 or later)
    year_node = evaluator.add_leaf(
        id=f"wine_{wine_index}_year_verify",
        desc=f"Verify the year is 2019 or later",
        parent=wine_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The year {wine_details.award_year} is 2019 or later.",
        node=year_node,
    )

    # Verify year from URL provenance
    year_url_node = evaluator.add_leaf(
        id=f"wine_{wine_index}_year_url_verify",
        desc=f"Verify the award year from source URLs",
        parent=wine_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The wine {wine_name} appeared in Wine Spectator's Top 10 list in the year {wine_details.award_year}",
        node=year_url_node,
        sources=wine_urls,
        additional_instruction="Verify that the source confirms this wine appeared in the stated year.",
    )

    # Verify ranking is valid (top 10)
    ranking_node = evaluator.add_leaf(
        id=f"wine_{wine_index}_ranking_verify",
        desc=f"Verify the ranking is in the top 10",
        parent=wine_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The ranking '{wine_details.ranking}' represents a position in the top 10.",
        node=ranking_node,
    )

    # Verify ranking from URL provenance
    ranking_url_node = evaluator.add_leaf(
        id=f"wine_{wine_index}_ranking_url_verify",
        desc=f"Verify the ranking from source URLs",
        parent=wine_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The wine {wine_name} achieved the ranking of {wine_details.ranking} in Wine Spectator's Top 10 list in the year {wine_details.award_year}",
        node=ranking_url_node,
        sources=wine_urls,
        additional_instruction="Verify that the source confirms this wine's specific ranking position.",
    )

    # Link verification - link MUST be provided per task requirements
    link_exists = evaluator.add_custom_node(
        result=bool(wine_details.link),
        id=f"wine_{wine_index}_link_exists",
        desc="Check if a direct Wine Spectator link is provided as required by task",
        parent=wine_node,
        critical=True
    )

    link_node = evaluator.add_leaf(
        id=f"wine_{wine_index}_link_verify",
        desc=f"Verify the link points to Wine Spectator Top 10 page",
        parent=wine_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"This page is to a Wine Spectator Top 10 Wines list page or an individual wine entry on Wine Spectator related to their Top 10 list in the year {wine_details.award_year}.",
        node=link_node,
        sources=[wine_details.link],
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

    # -------- 2. Extract information in steps ---------------------------- #
    # First, extract just the wine names
    wine_names_obj = await evaluator.extract(
        prompt=prompt_extract_wine_names(),
        template_class=WineNamesList,
        extraction_name="wine_names"
    )

    # For each wine name, extract its details and related URLs
    wines_data = []
    for wine_name in wine_names_obj.wine_names:
        # Extract details for this wine
        details = await evaluator.extract(
            prompt=prompt_extract_wine_details(wine_name),
            template_class=WineDetails,
            extraction_name=f"wine_details_{len(wines_data) + 1}"
        )

        # Extract URLs specifically related to this wine
        urls_obj = await evaluator.extract(
            prompt=prompt_extract_wine_urls(wine_name),
            template_class=WineSourceUrls,
            extraction_name=f"wine_urls_{len(wines_data) + 1}"
        )

        # Add main link to URLs if it exists and isn't already included
        if details.link and details.link not in urls_obj.urls:
            urls_obj.urls.append(details.link)

        wines_data.append((wine_name, details, urls_obj.urls))

    # -------- 3. Build verification tree --------------------------------- #
    # Create a node for all wines verification
    wines_node = evaluator.add_parallel(
        id="wines",
        desc="Verification of five Rhône wines from Wine Spectator's Top 10 since 2019",
    )

    # -------- 4. Pad to exactly 5 wines ---------------------------------- #
    while len(wines_data) < 5:
        wines_data.append(("", WineDetails(), []))

    # Verify all 5 wines
    for i in range(5):
        wine_name, wine_details, wine_urls = wines_data[i]
        await verify_single_wine(evaluator, wines_node, wine_name, wine_details, wine_urls, i)

    # -------- 5. Add custom info ----------------------------------------- #
    # Format the wine data for the info field
    formatted_wines = []
    for wine_name, details, urls in wines_data[:len(wine_names_obj.wine_names)]:  # Only format actual wines
        if wine_name:  # Skip empty entries
            formatted_wines.append({
                "name": wine_name,
                "region": details.region,
                "award_year": details.award_year,
                "ranking": details.ranking,
                "link": details.link,
                "related_urls": urls,
            })

    evaluator.add_custom_info({
        "extracted_wines": formatted_wines,
        "provided_wines": len(wine_names_obj.wine_names),
        "wines_to_verify": 5,
        "missing_wines": max(0, 5 - len(wine_names_obj.wine_names)),
    }, "wine_summary")

    # -------- 6. Return structured result -------------------------------- #
    return evaluator.get_summary()