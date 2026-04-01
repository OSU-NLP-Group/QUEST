import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "diseases_hospital"
TASK_DESCRIPTION = """
Find the statistics on major causes of death for people 65 and older in the US in 2002 (attach the url to the statistics) and list five top diseases. For each disease, please tell me which main specialty it belongs to (except Geriatrics) and find out which hospital in the US is the best for this specialty according to US News ranking.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #

class SourceURL(BaseModel):
    """A source URL with description"""
    url: Optional[str] = None
    description: Optional[str] = None

class StatisticsSource(BaseModel):
    """Source URL for the death statistics for people 65+ in US (2002)"""
    urls: List[SourceURL] = Field(default_factory=list)

class DiseaseNames(BaseModel):
    """List of disease names"""
    names: List[str] = Field(default_factory=list)

class DiseaseDetails(BaseModel):
    """Details for a specific disease"""
    specialty: Optional[str] = None
    top_hospital: Optional[str] = None
    source_urls: List[SourceURL] = Field(default_factory=list)

class ExtractedInfo(BaseModel):
    """All extracted information from the answer"""
    statistics_source: Optional[StatisticsSource] = None
    disease_names: List[str] = Field(default_factory=list)
    disease_details: Dict[str, DiseaseDetails] = Field(default_factory=dict)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #

def prompt_extract_statistics_source() -> str:
    return """
    Extract all URLs that provide statistics on major causes of death for people 65 and older in the US in 2002.
    
    Return:
    - urls: A list of source URLs, where each source has:
      - url: The URL to the source
      - description: A brief description of what this source contains
    
    If no URLs are provided in the answer, return an empty list.
    """

def prompt_extract_disease_names() -> str:
    return """
    Extract the names of the top 5 diseases mentioned as causes of death for people 65 and older in the US in 2002.
    
    Return a list of disease names in the order they appear in the answer.
    If fewer than 5 diseases are mentioned, extract only those that are provided.
    """

def prompt_extract_disease_details(disease_name: str) -> str:
    return f"""
    For the disease "{disease_name}" mentioned in the answer, extract:
    - specialty: The medical specialty associated with this disease (excluding Geriatrics)
    - top_hospital: The name of the top-ranked US hospital for this specialty according to US News
    - source_urls: A list of URLs mentioned in the answer that support information about this disease, its specialty, or the top hospital. For each URL include:
      - url: The URL itself
      - description: Brief description of what this URL contains (e.g., "disease statistics", "specialty information", "hospital ranking")
    
    If any of this information is missing, set the value to null for that specific field.
    For source_urls, if no URLs are provided, return an empty list.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #

async def verify_statistics_source(
    evaluator: Evaluator,
    parent_node,
    info: ExtractedInfo,
) -> None:
    """
    Verify that valid URLs are provided for the statistics on causes of death
    for people 65+ in the US in 2002, and that at least one URL actually contains this information.
    """
    statistics_node = evaluator.add_parallel(
        id="statistics_source_verification",
        desc="Verify that valid URL sources are provided for 2002 US death statistics for people 65+ and at least one contains the claimed information",
        parent=parent_node,
        critical=True,  # This is critical as it's a core requirement of the task
    )
    
    # Extract URLs from statistics source
    urls = []
    if info.statistics_source and info.statistics_source.urls:
        urls = [source.url for source in info.statistics_source.urls if source.url]
    
    # Add existence check
    exists_node = evaluator.add_custom_node(
        result=bool(urls),
        id="statistics_urls_exist",
        desc="Check if statistics URLs are provided",
        parent=statistics_node,
        critical=True
    )
    
    # Add verification node
    verification_node = evaluator.add_leaf(
        id="statistics_content_verification",
        desc="Verify URLs contain statistics on causes of death for people 65+ in US 2002",
        parent=statistics_node,
        critical=True
    )
    
    claim = "The provided URLs contain statistics on major causes of death for people 65 and older in the US in 2002"
    await evaluator.verify(
        claim=claim,
        node=verification_node,
        sources=urls,
        additional_instruction="Check if any of the webpages contain statistics or data on causes of death for people 65 and older in the US specifically for the year 2002. At least one page must have actual mortality statistics for this demographic group for 2002 (not just general information about diseases in elderly)."
    )

async def verify_diseases(
    evaluator: Evaluator,
    parent_node,
    info: ExtractedInfo,
) -> None:
    """
    Verify the diseases with all three criteria for each disease.
    """
    diseases_node = evaluator.add_parallel(
        id="diseases_verification",
        desc="Verify that the answer lists diseases from 2002 US death statistics for people 65+ and provides correct specialties and hospitals",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )
    
    # First verify that we have diseases
    has_diseases_node = evaluator.add_custom_node(
        result=bool(info.disease_names),
        id="has_diseases",
        desc="Verify that the answer attempts to list diseases as causes of death",
        parent=diseases_node,
        critical=True
    )
    
    # Get statistics sources for disease verification
    stats_urls = []
    if info.statistics_source and info.statistics_source.urls:
        stats_urls = [source.url for source in info.statistics_source.urls if source.url]
    
    # Ensure we have 5 diseases (pad with empty if needed)
    disease_list = list(info.disease_names)
    while len(disease_list) < 5:
        disease_list.append("")
    
    # Check each disease
    for i in range(5):
        disease_name = disease_list[i]
        details = info.disease_details.get(disease_name, DiseaseDetails()) if disease_name else DiseaseDetails()
        
        await verify_disease_complete(
            evaluator,
            diseases_node,
            disease_name,
            details,
            stats_urls,
            i + 1
        )

async def verify_disease_complete(
    evaluator: Evaluator,
    parent_node,
    disease_name: str,
    disease_details: DiseaseDetails,
    stats_urls: List[str],
    index: int
) -> None:
    """
    Verify a single disease with all three criteria.
    """
    disease_node = evaluator.add_parallel(
        id=f"disease_{index}_complete",
        desc=f"Complete verification for disease #{index}: {disease_name if disease_name else 'Missing'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )
    
    # Check if disease exists
    disease_exists_node = evaluator.add_custom_node(
        result=bool(disease_name),
        id=f"disease_{index}_exists",
        desc=f"Check if disease #{index} is provided",
        parent=disease_node,
        critical=True
    )
    
    # 1. Verify the disease is a top cause of death
    disease_validity_node = evaluator.add_leaf(
        id=f"disease_{index}_validity",
        desc=f"Verify that {disease_name if disease_name else 'the disease'} is a top cause of death",
        parent=disease_node,
        critical=True
    )
    
    # Collect all available URLs
    all_urls = list(stats_urls)
    if disease_details and disease_details.source_urls:
        all_urls.extend([source.url for source in disease_details.source_urls if source.url])
    
    claim = f"{disease_name} is one of the top causes of death for people 65 and older in the US in 2002"
    await evaluator.verify(
        claim=claim,
        node=disease_validity_node,
        sources=all_urls,
        additional_instruction="Check if this specific disease is explicitly mentioned as a top cause of death for elderly people (65+) in the US in 2002 statistics."
    )
    
    # 2. Verify specialty
    specialty_exists_node = evaluator.add_custom_node(
        result=bool(disease_details.specialty),
        id=f"specialty_{index}_exists",
        desc=f"Check if specialty is provided for disease #{index}",
        parent=disease_node,
        critical=True
    )
    
    specialty_node = evaluator.add_leaf(
        id=f"specialty_{index}_verification",
        desc=f"Verify specialty for {disease_name if disease_name else 'the disease'}",
        parent=disease_node,
        critical=True
    )
    
    specialty = disease_details.specialty or ""
    claim = f"'{specialty}' is a valid medical specialty (not Geriatrics) that is appropriate for treating {disease_name}"
    await evaluator.verify(
        claim=claim,
        node=specialty_node,
        sources=all_urls,
        additional_instruction="Verify that the given specialty is a legitimate medical specialty recognized in the medical field, and that it is NOT Geriatrics."
    )
    
    # 3. Verify hospital
    hospital_exists_node = evaluator.add_custom_node(
        result=bool(disease_details.top_hospital),
        id=f"hospital_{index}_exists",
        desc=f"Check if hospital is provided for disease #{index}",
        parent=disease_node,
        critical=True
    )
    
    hospital_node = evaluator.add_leaf(
        id=f"hospital_{index}_verification",
        desc=f"Verify top hospital for {specialty if specialty else 'the specialty'}",
        parent=disease_node,
        critical=True
    )
    
    hospital = disease_details.top_hospital or ""
    # Get hospital-specific URLs
    hospital_urls = []
    if disease_details and disease_details.source_urls:
        for source in disease_details.source_urls:
            if source.url and source.description and ("hospital" in source.description.lower() or "ranking" in source.description.lower()):
                hospital_urls.append(source.url)
    
    claim = f"{hospital} is ranked as the top US hospital for {specialty} according to US News rankings"
    await evaluator.verify(
        claim=claim,
        node=hospital_node,
        sources=hospital_urls if hospital_urls else all_urls,
        additional_instruction="Check if the webpage confirms that this hospital is ranked highly or as the best for this specific specialty according to US News rankings."
    )

# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate a single answer for the diseases_hospital task and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
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

    # -------- 2. Extract structured info from the answer ---------------- #
    # First extract the statistics source
    statistics_source = await evaluator.extract(
        prompt=prompt_extract_statistics_source(),
        template_class=StatisticsSource,
        extraction_name="statistics_source"
    )
    
    # Extract disease names first
    disease_names_result = await evaluator.extract(
        prompt=prompt_extract_disease_names(),
        template_class=DiseaseNames,
        extraction_name="disease_names"
    )
    
    disease_names = disease_names_result.names if disease_names_result.names else []
    
    # Then extract details for each disease
    disease_details = {}
    for name in disease_names:
        if name:  # Skip empty names
            details = await evaluator.extract(
                prompt=prompt_extract_disease_details(name),
                template_class=DiseaseDetails,
                extraction_name=f"disease_details_{name}"
            )
            disease_details[name] = details
    
    # Combine into final structure
    extracted_info = ExtractedInfo(
        statistics_source=statistics_source,
        disease_names=disease_names,
        disease_details=disease_details
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Verify the two main components of the task
    
    # 1. Verify statistics source URLs are provided and at least one contains the right information
    await verify_statistics_source(evaluator, root, extracted_info)
    
    # 2. Verify diseases with complete verification for each
    await verify_diseases(evaluator, root, extracted_info)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()