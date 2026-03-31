import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "plants_italy"
TASK_DESCRIPTION = """
Please identify two plant species with distinct genera that are originally native to regions in Africa, Asia, or the Americas, and have later been introduced to Italy. What are their scientific names? Then for each species, provide a link to its distribution information on Plants of the World Online. Please also provide links to Wikipedia articles about the genera of the two plant species.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PlantSpecies(BaseModel):
    scientific_name: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    native_region: Optional[str] = None
    powo_url: Optional[str] = None
    wikipedia_genus_url: Optional[str] = None


class ExtractedPlantInfo(BaseModel):
    plants: List[PlantSpecies] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_plants() -> str:
    return """
    Extract information about plant species mentioned in the answer. For each plant species, extract:
    
    1. The scientific name (both genus and species, e.g., "Opuntia ficus-indica")
    2. The genus name (e.g., "Opuntia")
    3. The species name (e.g., "ficus-indica")
    4. The native region mentioned (e.g., "Americas", "Africa", "Asia")
    5. The URL to Plants of the World Online (POWO) for this species
    6. The URL to the Wikipedia article about the genus
    
    Extract this information for all plant species mentioned in the answer, even if more than two are provided.
    If any of these pieces of information is not explicitly provided in the answer, set that field to null.
    """


# --------------------------------------------------------------------------- #
# Plant verification functions                                                #
# --------------------------------------------------------------------------- #
async def verify_plant(
    evaluator: Evaluator,
    parent_node,
    plant: PlantSpecies,
    index: int,
) -> None:
    """
    Verify all aspects of a single plant species.
    """
    # Create parallel parent node for this plant
    plant_node = evaluator.add_parallel(
        id=f"plant_{index}",
        desc=f"Verification of all requirements for plant species {index}: {plant.scientific_name if plant.scientific_name else 'Missing plant'}",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit
    )
    
    # 1. Overall completeness check (gates all subsequent verifications)
    completeness_node = evaluator.add_custom_node(
        result=bool(
            plant.scientific_name and 
            plant.genus and 
            plant.wikipedia_genus_url and 
            plant.powo_url
        ),
        id=f"plant_{index}_completeness",
        desc=f"Check if all required information is provided for plant {index}",
        parent=plant_node,
        critical=True
    )
    
    # 2. Verify Wikipedia genus URL
    wikipedia_node = evaluator.add_leaf(
        id=f"plant_{index}_wikipedia_url",
        desc=f"Verify that the Wikipedia URL correctly links to the genus page",
        parent=plant_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The webpage is a Wikipedia page about the plant genus '{plant.genus}'.",
        node=wikipedia_node,
        sources=plant.wikipedia_genus_url,
        additional_instruction=f"Check if the webpage is a Wikipedia page about the plant genus '{plant.genus}'. The genus name should be prominently displayed on the page."
    )
    
    # 3. Verify POWO URL
    powo_node = evaluator.add_leaf(
        id=f"plant_{index}_powo_url",
        desc=f"Verify that the POWO URL correctly links to the species page",
        parent=plant_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The webpage is a Plants of the World Online (POWO) page of the plant species '{plant.scientific_name}', which shows the distribution information of plant species '{plant.scientific_name}'.",
        node=powo_node,
        sources=plant.powo_url,
        additional_instruction=f"Check if the webpage is POWO page of the plant species '{plant.scientific_name}'. The scientific name should be prominently displayed on the page."
    )
    
    # 4. Verify native region using POWO URL
    native_region_node = evaluator.add_leaf(
        id=f"plant_{index}_native_region",
        desc=f"Verify that plant {index} is native to Africa, Asia, or the Americas",
        parent=plant_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The plant species '{plant.scientific_name}' is originally native to regions in Africa, Asia, or the Americas.",
        node=native_region_node,
        sources=plant.powo_url,
        additional_instruction="Look for distribution information on the webpage. Verify that the plant is native to regions in Africa, Asia, or the Americas."
    )
    
    # 5. Verify introduced to Italy using POWO URL
    italy_node = evaluator.add_leaf(
        id=f"plant_{index}_introduced_to_italy",
        desc=f"Verify that plant {index} has been introduced to Italy",
        parent=plant_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The plant species '{plant.scientific_name}' has been introduced to Italy.",
        node=italy_node,
        sources=plant.powo_url,
        additional_instruction="Look for distribution information on the webpage. Verify that the plant has been introduced to or is present in Italy. It may be listed as 'introduced', 'naturalized', 'alien', or 'non-native' in Italy."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer to the plants_italy task and return a structured result dictionary.
    
    This evaluation checks if:
    1. The answer identified at least two plant species with distinct genera
    2. Each plant has a proper scientific name
    3. Wikipedia URLs are provided for each plant's genus
    4. POWO URLs are provided for each plant's distribution
    5. The plants are native to Africa, Asia, or the Americas (not Europe/Italy)
    6. The plants have been introduced to Italy
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
        default_model=model
    )
    
    # -------- 2. Extract structured info from the answer ---------------- #
    parsed_info = await evaluator.extract(
        prompt=prompt_extract_plants(),
        template_class=ExtractedPlantInfo,
        extraction_name="plant_info"
    )
    
    # -------- 3. Deduplicate plants by genus (keep first occurrence) ----- #
    distinct_genera_plants = []
    seen_genera = set()
    
    for plant in parsed_info.plants:
        if plant.genus and plant.genus not in seen_genera:
            seen_genera.add(plant.genus)
            distinct_genera_plants.append(plant)
    
    # Pad to ensure we have exactly 2 plants (using empty PlantSpecies for missing ones)
    while len(distinct_genera_plants) < 2:
        distinct_genera_plants.append(PlantSpecies())
    
    # Limit to first two plants
    plants_to_verify = distinct_genera_plants[:2]
    
    # -------- 4. Add custom info about distinct genera ----------------- #
    evaluator.add_custom_info(
        {"distinct_genera_plants": [p.dict() for p in plants_to_verify]},
        "distinct_genera_info"
    )
    
    # -------- 5. Build verification tree -------------------------------- #
    # Verify each plant with distinct genus
    for i, plant in enumerate(plants_to_verify, 1):
        await verify_plant(evaluator, root, plant, i)
    
    # -------- 6. Return structured result ------------------------------- #
    return evaluator.get_summary()