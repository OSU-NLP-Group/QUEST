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
TASK_ID = "cleveland_museum"
TASK_DESCRIPTION = """
Identify five ancient Chinese objects from the Cleveland Museum of Art's collection that dated entirely to 2000 BCE or earlier, as listed on the museum's official website. For each object, provide its name, period, estimated date range, overall dimensions, and current location within the museum. If an object is not currently on display, please indicate that as well.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for information extraction                                      #
# --------------------------------------------------------------------------- #
class ObjectName(BaseModel):
    """Model for object name extraction."""
    name: str = None


class ObjectNames(BaseModel):
    """Container for all extracted object names."""
    objects: List[ObjectName] = Field(default_factory=list)


class ArtObject(BaseModel):
    """Model for a single ancient Chinese art object."""
    name: Optional[str] = None
    period: Optional[str] = None
    date_range: Optional[str] = None
    dimensions: Optional[str] = None
    location: Optional[str] = None
    on_display: Optional[bool] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_object_names() -> str:
    return """
    Extract the names of all ancient Chinese objects mentioned in the answer. 
    
    Just extract the name or title of each object as it appears in the answer. 
    If fewer than five objects are mentioned, extract only those that are mentioned.
    
    Return the extracted names in a list under the "objects" field, with each object having a "name" field.
    """


def prompt_extract_object_details(object_name: str) -> str:
    return f"""
    Extract detailed information about the ancient Chinese object named "{object_name}" from the answer.
    
    Extract the following details:
    1. name: The name or title of the object (should be "{object_name}" or very similar)
    2. period: The historical period/dynasty the object belongs to
    3. date_range: The estimated date range (must be 2000 BCE or earlier)
    4. dimensions: The physical dimensions of the object
    5. location: The location within the museum where the object is displayed. If the object is not on display, this should be null.
    6. on_display: Whether the object is currently on display (true) or not (false). If a specific location within the museum is provided, then this field should be true. If the answer explicitly mentions it is not on display now, this field should be false.
    7. source_urls: Any URLs mentioned in the answer that point to the Cleveland Museum of Art's website (clevelandart.org) or any other source used to find information about this specific object
    
    Return null for any field that is not mentioned in the answer. If the answer doesn't clearly indicate whether an object is on display, return null for the on_display field.
    
    Note: The location field should only be populated if the object is currently on display with a specific location. If the object is not on display, both location should be null and on_display should be false.
    
    For source_urls, make sure to extract complete and valid URLs. If a URL is mentioned without a protocol, prepend 'http://'.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_object_has_official_source(
    evaluator: Evaluator,
    parent_node,
    object_index: int,
    art_object: ArtObject,
) -> None:
    """
    Verify that the object has at least one official Cleveland Museum of Art source URL.
    """
    # Direct check without wrapper parent node
    official_source_check = evaluator.add_custom_node(
        result=bool(art_object.source_urls and any("clevelandart.org" in url.lower() for url in art_object.source_urls)),
        id=f"object_{object_index+1}_has_official_source",
        desc=f"Object #{object_index+1} has at least one official Cleveland Museum of Art source URL",
        parent=parent_node,
        critical=True
    )


async def verify_object_completeness(
    evaluator: Evaluator,
    parent_node,
    object_index: int,
    art_object: ArtObject,
) -> None:
    """
    Verify that an object has all the required information fields.
    """
    # Direct check without wrapper parent node
    completeness_check = evaluator.add_custom_node(
        result=bool(
            art_object.name is not None and
            art_object.period is not None and
            art_object.date_range is not None and
            art_object.dimensions is not None and
            art_object.on_display is not None and
            # Updated logic: location is required only if on_display is True
            (art_object.location is not None if art_object.on_display else True)
        ),
        id=f"object_{object_index+1}_completeness",
        desc=f"Object #{object_index+1} has all required information fields (location required only if on display)",
        parent=parent_node,
        critical=True
    )


async def verify_object_date(
    evaluator: Evaluator,
    parent_node,
    object_index: int,
    art_object: ArtObject,
) -> None:
    """
    Verify that the object dates to 2000 BCE or earlier.
    """
    # Direct verification without existence check (completeness already checks date_range)
    date_verification_node = evaluator.add_leaf(
        id=f"object_{object_index+1}_date_verification",
        desc=f"Object #{object_index+1} dates to 2000 BCE or earlier",
        parent=parent_node,
        critical=True,
    )
    
    # Use empty string if date_range is None to avoid "None" in the claim
    date_range = art_object.date_range or "[no date range provided]"
    claim = f"The date range '{date_range}' indicates that the object is from 2000 BCE or earlier."
    
    await evaluator.verify(
        claim=claim,
        node=date_verification_node,
        additional_instruction="""
        Verify if the date range provided indicates that the object dates entirely to 2000 BCE or earlier.
        - If the date range includes any years after 2000 BCE, the verification should fail.
        - BCE dates (Before Common Era) are older than CE dates.
        - Dates like "c. 2500 BCE" or "ca. 3000-2500 BCE" would qualify as entirely 2000 BCE or earlier.
        - Dynasties or periods known to be entirely 2000 BCE or earlier (like Neolithic period in China) would qualify.
        """
    )


async def verify_chinese_origin(
    evaluator: Evaluator,
    parent_node,
    object_index: int,
    art_object: ArtObject,
) -> None:
    """
    Verify that the object is indeed from ancient China.
    """
    # Direct verification without existence check (has_official_source already checks source_urls)
    chinese_origin_node = evaluator.add_leaf(
        id=f"object_{object_index+1}_chinese_origin_verification",
        desc=f"Object #{object_index+1} is from ancient China",
        parent=parent_node,
        critical=True,
    )
    
    claim = f"The object '{art_object.name or 'Unknown'}' is from ancient China."
    await evaluator.verify(
        claim=claim,
        node=chinese_origin_node,
        sources=art_object.source_urls,
        additional_instruction="""
        Verify that the object is specifically identified as being from China, Chinese in origin, 
        or associated with a Chinese dynasty or period. The object should be categorized under Chinese art 
        or explicitly stated to be from China.
        """
    )


async def verify_object_provenance(
    evaluator: Evaluator,
    parent_node,
    object_index: int,
    art_object: ArtObject,
) -> None:
    """
    Verify that the object information is correctly sourced from Cleveland Museum of Art.
    """
    # Direct verification without existence check (has_official_source already checks source_urls)
    provenance_node = evaluator.add_leaf(
        id=f"object_{object_index+1}_provenance_verification",
        desc=f"Object #{object_index+1} information is correctly sourced from Cleveland Museum of Art",
        parent=parent_node,
        critical=True,
    )
    
    # Construct a claim containing all the object information
    object_info = f"Object: {art_object.name or 'Unknown'}\n"
    if art_object.period:
        object_info += f"Period: {art_object.period}\n"
    if art_object.date_range:
        object_info += f"Date Range: {art_object.date_range}\n"
    if art_object.dimensions:
        object_info += f"Dimensions: {art_object.dimensions}\n"
    if art_object.location:
        object_info += f"Location: {art_object.location}\n"
    if art_object.on_display is not None:
        object_info += f"On Display: {'Yes' if art_object.on_display else 'No'}\n"
    
    await evaluator.verify(
        claim=object_info,
        node=provenance_node,
        sources=art_object.source_urls,
        additional_instruction="""
        When verifying, check if the provided information matches what's on the Cleveland Museum of Art's website. 
        Specifically, the object information (name, date, period, dimensions, location) should match what's on the museum's website
        
        Note that the website may format dimensions slightly differently or provide additional details, but the core 
        information should match. The on_display status might be indicated by phrases like "Not on View" or similar.
        """
    )


async def verify_object(
    evaluator: Evaluator,
    parent_node,
    object_index: int,
    art_object: Optional[ArtObject],
) -> None:
    """
    Verify all aspects of a single art object using parallel verification.
    If the object is None (missing), create a placeholder node.
    """
    # Create parent node for this object - using PARALLEL as per feedback
    object_node = evaluator.add_parallel(
        id=f"object_{object_index+1}",
        desc=f"Object #{object_index+1}: '{art_object.name if art_object and art_object.name else 'Unknown'}' meets all requirements",
        parent=parent_node,
        critical=False,
    )
    
    # Single existence check directly on object_node
    existence_check = evaluator.add_custom_node(
        result=art_object is not None,
        id=f"object_{object_index+1}_exists",
        desc=f"Object #{object_index+1} exists in the answer",
        parent=object_node,
        critical=True
    )
    
    # If object doesn't exist, all other verifications will be gated by the existence check
    # Still create the verification nodes to maintain tree structure
    await verify_object_has_official_source(evaluator, object_node, object_index, art_object or ArtObject())
    await verify_object_completeness(evaluator, object_node, object_index, art_object or ArtObject())
    await verify_object_date(evaluator, object_node, object_index, art_object or ArtObject())
    await verify_chinese_origin(evaluator, object_node, object_index, art_object or ArtObject())
    await verify_object_provenance(evaluator, object_node, object_index, art_object or ArtObject())


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

    # -------- 2. Extract structured info from the answer ----------------- #
    # First extract the object names
    object_names = await evaluator.extract(
        prompt=prompt_extract_object_names(),
        template_class=ObjectNames,
        extraction_name="object_names"
    )
    
    # Then extract detailed information for each object
    art_objects = []
    for i, obj in enumerate(object_names.objects[:5]):  # Limit to max 5 objects
        art_object = await evaluator.extract(
            prompt=prompt_extract_object_details(obj.name),
            template_class=ArtObject,
            extraction_name=f"art_object_{i+1}"
        )
        # If name is somehow missing in the detailed extraction, use the name from the first extraction
        if art_object.name is None:
            art_object.name = obj.name
        art_objects.append(art_object)

    # -------- 3. Build verification tree -------------------------------- #
    # Process each object (or placeholder if fewer than 5 objects)
    for i in range(5):
        if i < len(art_objects):
            await verify_object(evaluator, root, i, art_objects[i])
        else:
            await verify_object(evaluator, root, i, None)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()