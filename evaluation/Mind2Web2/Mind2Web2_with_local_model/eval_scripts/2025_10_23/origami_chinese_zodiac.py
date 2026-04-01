import asyncio
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "origami_chinese_zodiac"
TASK_DESCRIPTION = """
Recommend origami tutorials for each of the 12 Chinese Zodiac animals. Each tutorial must be photo-based or diagram-based, with clearly numbered steps (no videos). Provide exactly one tutorial per animal. I'm very lazy and don't want to deal with super lengthy tutorials, so make sure none of them exceeds 30 steps.
"""

# The 12 Chinese Zodiac animals
ZODIAC_ANIMALS = [
    "rat", "ox", "tiger", "rabbit", "dragon", "snake", 
    "horse", "goat", "monkey", "rooster", "dog", "pig"
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                      #
# --------------------------------------------------------------------------- #
class OrigamiTutorial(BaseModel):
    """Model for a single origami tutorial."""
    animal: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None


class OrigamiTutorials(BaseModel):
    """Model for all extracted origami tutorials."""
    tutorials: List[OrigamiTutorial] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                         #
# --------------------------------------------------------------------------- #
def prompt_extract_tutorials() -> str:
    return """
    Extract all the origami tutorials for Chinese Zodiac animals mentioned in the answer. For each tutorial:
    1. Extract the animal it represents (e.g., "rat", "ox", etc.)
    2. Extract the URL of the tutorial
    3. Extract a brief description or name of the tutorial if available

    Format the information into the OrigamiTutorials schema with a list of tutorials, each containing 'animal', 'url', and 'description' fields.
    
    When extracting animals, normalize the names to match the standard Chinese zodiac animals: rat, ox, tiger, rabbit, dragon, snake, horse, goat, monkey, rooster, dog, and pig. For example, if the answer mentions "sheep" instead of "goat", normalize it to "goat" in your extraction.
    
    If an animal isn't clearly specified or can't be matched to one of the 12 zodiac animals, set the animal field to null.
    
    If a URL isn't provided for a tutorial, set the url field to null.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_animal_tutorial(
    evaluator: Evaluator,
    parent_node,
    animal: str,
    tutorial: OrigamiTutorial,
    index: int,
) -> None:
    """
    Verify that the tutorial for a specific zodiac animal meets all requirements:
    1. Has a valid tutorial for this animal
    2. Tutorial is photo/diagram-based (not video)
    3. Tutorial does not exceed 30 steps
    """
    # Create a parallel node for this animal
    animal_node = evaluator.add_parallel(
        id=f"animal_{index}_{animal}",
        desc=f"Verification of origami tutorial for {animal}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit across animals
    )
    
    # Check if tutorial exists and has a URL (critical existence check)
    tutorial_exists = evaluator.add_custom_node(
        result=bool(tutorial.url),
        id=f"{animal}_tutorial_exists",
        desc=f"Check if a tutorial URL was provided for {animal}",
        parent=animal_node,
        critical=True  # Critical to gate subsequent verifications
    )
    
    # Step 1: Verify the URL contains a tutorial for this animal
    has_tutorial_node = evaluator.add_leaf(
        id=f"{animal}_has_valid_tutorial",
        desc=f"Verifying that the URL contains an origami tutorial for {animal}",
        parent=animal_node,
        critical=True,
    )
    
    claim = f"The webpage contains an origami tutorial for making a {animal}."
    
    additional_instruction = f"""When verifying if this is a {animal} tutorial, be flexible with animal naming:
- Accept common synonyms and variations of the animal name
- For example: 'mouse' is acceptable for 'rat', 'bull/cow/cattle' for 'ox', 'sheep/ram' for 'goat', 'bunny/hare' for 'rabbit', 'cock/cockerel/chicken' for 'rooster', etc.
- Focus on whether the tutorial is clearly for making an origami representation of the aminal {animal}, regardless of the exact term used
- The key is that it should be recognizable as the animal, not necessarily using the exact terminology"""
    
    await evaluator.verify(
        claim=claim,
        node=has_tutorial_node,
        sources=tutorial.url,
        additional_instruction=additional_instruction,
    )
    
    # Step 2: Verify the tutorial is photo/diagram-based with no video
    photo_based_node = evaluator.add_leaf(
        id=f"{animal}_is_photo_based",
        desc=f"Verifying the {animal} tutorial is photo/diagram-based and not video-based",
        parent=animal_node,
        critical=True,
    )
    
    claim = f"The {animal} origami tutorial is photo or diagram-based, with clearly visible images or diagrams showing the folding steps, and is NOT primarily a video tutorial."
    await evaluator.verify(
        claim=claim,
        node=photo_based_node,
        sources=tutorial.url,
    )
    
    # Step 3: Verify the tutorial has numbered steps and doesn't exceed 30 steps
    step_count_node = evaluator.add_leaf(
        id=f"{animal}_step_count_valid",
        desc=f"Verifying the {animal} tutorial has clearly numbered steps and doesn't exceed 30 steps",
        parent=animal_node,
        critical=True,
    )
    
    claim = f"The {animal} origami tutorial has clearly numbered steps, and the total number of steps does NOT exceed 30."
    await evaluator.verify(
        claim=claim,
        node=step_count_node,
        sources=tutorial.url,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client,
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
    
    # Initialize evaluator with parallel strategy for root
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

    # -------- 2. Extract all tutorials from the answer -------------------- #
    tutorials = await evaluator.extract(
        prompt=prompt_extract_tutorials(),
        template_class=OrigamiTutorials,
        extraction_name="origami_tutorials"
    )

    # -------- 3. Build verification tree --------------------------------- #
    # Create a mapping of animals to tutorials for easy lookup
    animal_to_tutorial = {}
    for t in tutorials.tutorials:
        if t.animal and t.animal.lower() in [a.lower() for a in ZODIAC_ANIMALS]:
            animal_to_tutorial[t.animal.lower()] = t
    
    # Pad missing animals with empty tutorials
    tutorial_list = []
    for animal in ZODIAC_ANIMALS:
        if animal.lower() in animal_to_tutorial:
            tutorial_list.append(animal_to_tutorial[animal.lower()])
        else:
            # Create empty tutorial for missing animals
            tutorial_list.append(OrigamiTutorial(animal=animal, url=None, description=None))
    
    # Verify each animal's tutorial using unified logic
    for i, (animal, tutorial) in enumerate(zip(ZODIAC_ANIMALS, tutorial_list)):
        await verify_animal_tutorial(evaluator, root, animal, tutorial, i)

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()