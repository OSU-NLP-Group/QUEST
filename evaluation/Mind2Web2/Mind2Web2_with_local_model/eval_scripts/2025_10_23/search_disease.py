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
TASK_ID = "search_disease"
TASK_DESCRIPTION = """
My 1-year-old child has a fever and skin rash. Identify three illnesses in young children that match these symptoms (they may include additional symptoms). 
For each illness, provide a list of symptoms by combining information from two medical sources (chosen only from the CDC, Mayo Clinic, WebMD and Merck Manual). Include direct links to the exact pages used for verification.
Clearly determine whether a vaccine for this illness is available in the United States. If a vaccine is available, explicitly state this and provide a direct link to a CDC webpage with information about this vaccine. If no vaccine is available, state this explicitly.
"""

JUDGE_MODEL = "o4-mini"

# Valid medical sources
VALID_SOURCE_NAMES = ["CDC", "Mayo Clinic", "WebMD", "Merck Manual"]

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class VaccineInfo(BaseModel):
    available: Optional[bool] = None
    cdc_url: Optional[str] = None
    verification_urls: List[str] = Field(default_factory=list)  # URLs that support the vaccine availability claim


class IllnessDetailsRaw(BaseModel):
    """Raw extraction with all URLs"""
    combined_symptoms: List[str] = Field(default_factory=list)
    all_source_urls: List[str] = Field(default_factory=list)  # ALL medical source URLs mentioned
    vaccine_info: VaccineInfo = Field(default_factory=VaccineInfo)


class IllnessDetails(BaseModel):
    combined_symptoms: List[str] = Field(default_factory=list)
    source_url_1: Optional[str] = None
    source_url_2: Optional[str] = None
    vaccine_info: VaccineInfo = Field(default_factory=VaccineInfo)


class Illness(BaseModel):
    name: Optional[str] = None
    combined_symptoms: List[str] = Field(default_factory=list)
    source_url_1: Optional[str] = None
    source_url_2: Optional[str] = None
    vaccine_info: VaccineInfo = Field(default_factory=VaccineInfo)


class IllnessNames(BaseModel):
    illness_names: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def get_source_domain(url: str) -> Optional[str]:
    """Extract the medical source from a URL."""
    if not url:
        return None
    
    url_lower = url.lower()
    if "cdc.gov" in url_lower:
        return "cdc"
    elif "mayoclinic" in url_lower:
        return "mayoclinic"
    elif "webmd" in url_lower:
        return "webmd"
    elif "merck" in url_lower:
        return "merck"
    else:
        return None


def select_two_different_sources(urls: List[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Select two URLs from different medical sources.
    Returns (url1, url2) or (None, None) if not possible.
    """
    if not urls:
        return None, None
    
    # Group URLs by source
    urls_by_source = {}
    for url in urls:
        source = get_source_domain(url)
        if source:
            if source not in urls_by_source:
                urls_by_source[source] = []
            urls_by_source[source].append(url)
    
    # If we have at least 2 different sources, pick one URL from each
    if len(urls_by_source) >= 2:
        sources = list(urls_by_source.keys())
        return urls_by_source[sources[0]][0], urls_by_source[sources[1]][0]
    
    # If we only have one source, return what we have
    elif len(urls_by_source) == 1:
        source = list(urls_by_source.keys())[0]
        source_urls = urls_by_source[source]
        if len(source_urls) >= 2:
            return source_urls[0], source_urls[1]
        elif len(source_urls) == 1:
            return source_urls[0], None
    
    return None, None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_illness_names() -> str:
    return """
    Extract the names of three illnesses mentioned in the answer that:
    1. Affect young children
    2. Have both fever and rash as symptoms

    Return a list of exactly three illness names as they appear in the answer.
    If fewer than three illnesses are mentioned, return only those found.
    """


def prompt_extract_illness_details(illness_name: str) -> str:
    return f"""
    For the illness "{illness_name}", extract the following information from the answer:

    1. Combined symptom list: Extract all symptoms mentioned for this illness (may be a unified list from both sources)
    2. ALL source URLs: Extract ALL medical source URLs mentioned for this illness (not just the first two)
       - Include every URL from CDC, Mayo Clinic, WebMD, or Merck Manual that relates to this illness
       - Extract them in the order they appear in the answer
    3. Vaccine information:
       - Whether a vaccine is available (true/false)
       - If vaccine IS available: extract the CDC URL for vaccine information (cdc_url)
       - If vaccine is NOT available: extract any URLs provided that support this claim (verification_urls)
       - Note: verification_urls should be used for supporting "no vaccine available" claims

    Important:
    - Extract symptoms exactly as listed in the answer
    - Extract ALL medical source URLs, not just the first two mentioned
    - Medical source URLs must be from CDC, Mayo Clinic, WebMD, or Merck Manual
    - For vaccine info, extract the explicit statement about availability
    - Return null for any field not clearly provided
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_all_symptoms(
    evaluator: Evaluator,
    parent_node,
    illness_name: str,
    symptoms: List[str],
    source_urls: List[str],
    illness_index: int,
) -> None:
    """Verify all symptoms are supported by at least one source using a for loop."""
    
    # Track verification results for each symptom
    all_verified = True
    unsupported_symptoms = []
    
    # Verify each symptom
    if source_urls:
        for symptom in symptoms:
            claim = f"For the illness {illness_name}, the symptom '{symptom}' is mentioned or described on at least one of the provided medical webpages"
            
            # Call verify without assigning to a node (just get the result)
            is_supported = await evaluator.verify(
                claim=claim,
                node=None,  # Don't create individual nodes
                sources=source_urls,
                additional_instruction="Verify that this specific symptom is mentioned in relation to the illness on at least one of the webpages. Consider medical synonyms and related terms when matching symptoms."
            )
            
            if not is_supported:
                all_verified = False
                unsupported_symptoms.append(symptom)
    
        # Create a single node with the AND result
        if unsupported_symptoms:
            unsupported_str = ", ".join(unsupported_symptoms)
            description = f"Check if all symptoms for illness {illness_index} are supported (Failed - unsupported: {unsupported_str})"
        else:
            description = f"Check if all symptoms for illness {illness_index} are supported (All verified)"
    else:
        # No sources provided, cannot verify any symptoms
        all_verified = False
        description = f"Check if all symptoms for illness {illness_index} are supported (Failed - no sources provided)"
    
    evaluator.add_custom_node(
        result=all_verified,
        id=f"illness_{illness_index}_all_symptoms_verified",
        desc=description,
        parent=parent_node,
        critical=True
    )


async def verify_illness(
    evaluator: Evaluator,
    parent_node,
    illness: Illness,
    illness_index: int,
) -> None:
    """Verify a single illness with all its requirements."""
    
    illness_node = evaluator.add_sequential(
        id=f"illness_{illness_index}",
        desc=f"Illness {illness_index}: {illness.name or 'Unknown'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # 1. Check illness completeness
    completeness_node = evaluator.add_custom_node(
        result=(
            illness.name is not None and illness.name.strip() != "" and
            illness.source_url_1 is not None and
            illness.source_url_2 is not None and
            len(illness.combined_symptoms) > 0 and
            illness.vaccine_info.available is not None
        ),
        id=f"illness_{illness_index}_completeness",
        desc=f"Check if illness {illness_index} has all required information",
        parent=illness_node,
        critical=True
    )

    # 2. Verify symptoms include fever and rash using LLM
    symptoms_check_node = evaluator.add_leaf(
        id=f"illness_{illness_index}_symptoms_check",
        desc=f"Check if illness {illness_index} symptoms include both fever and rash",
        parent=illness_node,
        critical=True
    )

    symptoms_str = ", ".join(illness.combined_symptoms)
    claim = f"Given that {illness.name} has the following symptoms: [{symptoms_str}], this list of symptoms includes both (1) fever or similar (elevated temperature, pyrexia, high temperature, etc.) AND (2) rash or similar skin manifestation (skin rash, eruption, exanthem, skin lesions, etc.)"
    
    await evaluator.verify(
        claim=claim,
        node=symptoms_check_node,
        additional_instruction="Analyze the provided symptom list to determine if it contains both fever-related symptoms AND rash/skin-related symptoms. Consider medical synonyms and related terms. The symptoms must include BOTH conditions to pass."
    )

    # 3. Verify sources are valid and different using LLM
    sources_validity_node = evaluator.add_leaf(
        id=f"illness_{illness_index}_sources_validity",
        desc=f"Check if illness {illness_index} uses two different valid medical sources",
        parent=illness_node,
        critical=True
    )

    valid_sources_str = ", ".join(VALID_SOURCE_NAMES)
    claim = f"The two URLs '{illness.source_url_1}' and '{illness.source_url_2}' meet these requirements: (1) Both URLs are from the allowed medical sources ({valid_sources_str}), AND (2) The two URLs are from different medical organizations (not just different pages from the same organization)"
    
    await evaluator.verify(
        claim=claim,
        node=sources_validity_node,
        additional_instruction=f"Verify that both URLs are from the allowed medical sources: {valid_sources_str}. Then verify they are from DIFFERENT organizations. For example, two CDC.gov URLs would NOT be different sources, but a CDC.gov URL and a MayoClinic.org URL would be different sources. Consider domain variants (e.g., mayoclinic.org and mayoclinic.com are the same organization). If either URL is not from an allowed source, or if both URLs are from the same organization, this should fail."
    )

    # 4. Verify illness is mentioned on both sources
    illness_mentioned_node = evaluator.add_parallel(
        id=f"illness_{illness_index}_mentioned",
        desc=f"Verify illness {illness_index} is mentioned on both sources",
        parent=illness_node,
        critical=False
    )

    # Check source 1
    source1_mention_node = evaluator.add_leaf(
        id=f"illness_{illness_index}_source1_mention",
        desc=f"Verify {illness.name} is mentioned on source 1",
        parent=illness_mentioned_node,
        critical=True
    )
    
    claim = f"The webpage contains information about {illness.name}"
    await evaluator.verify(
        claim=claim,
        node=source1_mention_node,
        sources=illness.source_url_1,
        additional_instruction="Verify that this illness is discussed on the webpage, preferably in the context of young children or pediatric cases."
    )

    # Check source 2
    source2_mention_node = evaluator.add_leaf(
        id=f"illness_{illness_index}_source2_mention",
        desc=f"Verify {illness.name} is mentioned on source 2",
        parent=illness_mentioned_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=claim,
        node=source2_mention_node,
        sources=illness.source_url_2,
        additional_instruction="Verify that this illness is discussed on the webpage, preferably in the context of young children or pediatric cases."
    )

    # 5. Verify all symptoms using a for loop
    valid_source_urls = [url for url in [illness.source_url_1, illness.source_url_2] if url is not None]

    await verify_all_symptoms(
        evaluator,
        illness_node,
        illness.name,
        illness.combined_symptoms,
        valid_source_urls,
        illness_index
    )

    # 6. Verify vaccine information
    vaccine_node = evaluator.add_parallel(
        id=f"illness_{illness_index}_vaccine",
        desc=f"Vaccine information for illness {illness_index}",
        parent=illness_node,
        critical=False
    )

    # Check if vaccine information is complete based on availability
    if illness.vaccine_info.available is True:
        # Vaccine is available - need CDC URL
        vaccine_completeness_node = evaluator.add_custom_node(
            result=(illness.vaccine_info.cdc_url is not None),
            id=f"illness_{illness_index}_vaccine_completeness",
            desc=f"Check if CDC vaccine URL is provided for illness {illness_index} (vaccine available)",
            parent=vaccine_node,
            critical=True
        )
        
        # Verify CDC URL if provided
        if illness.vaccine_info.cdc_url:
            cdc_url_node = evaluator.add_leaf(
                id=f"illness_{illness_index}_cdc_url_verification",
                desc=f"Verify CDC vaccine information URL for illness {illness_index}",
                parent=vaccine_node,
                critical=True
            )

            claim = f"The CDC webpage contains vaccine information for {illness.name}"
            await evaluator.verify(
                claim=claim,
                node=cdc_url_node,
                sources=illness.vaccine_info.cdc_url,
                additional_instruction="Verify that this is a CDC webpage (cdc.gov) that contains vaccine information specifically related to the illness. The page should discuss vaccination for this illness."
            )
    
    elif illness.vaccine_info.available is False:
        # Vaccine is NOT available - need verification URLs
        vaccine_completeness_node = evaluator.add_custom_node(
            result=(len(illness.vaccine_info.verification_urls) > 0),
            id=f"illness_{illness_index}_vaccine_completeness",
            desc=f"Check if verification URLs are provided for illness {illness_index} (no vaccine available)",
            parent=vaccine_node,
            critical=True
        )
        
        # Verify the "no vaccine" claim if URLs provided
        if illness.vaccine_info.verification_urls:
            no_vaccine_node = evaluator.add_leaf(
                id=f"illness_{illness_index}_no_vaccine_verification",
                desc=f"Verify no vaccine is available for illness {illness_index}",
                parent=vaccine_node,
                critical=True
            )

            claim = f"The webpage confirms that no vaccine is currently available for {illness.name} in the United States"
            await evaluator.verify(
                claim=claim,
                node=no_vaccine_node,
                sources=illness.vaccine_info.verification_urls,
                additional_instruction="Verify that at least one of the provided webpages explicitly states or strongly implies that no vaccine is available for this illness. Look for statements about lack of vaccination, no available vaccine, or treatment-only approaches."
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

    # -------- 2. Extract illness names first ---------------------------- #
    illness_names_result = await evaluator.extract(
        prompt=prompt_extract_illness_names(),
        template_class=IllnessNames,
        extraction_name="illness_names"
    )

    # -------- 3. Extract details for each illness ----------------------- #
    illnesses = []
    
    # Process found illnesses
    for illness_name in illness_names_result.illness_names[:3]:  # Take at most 3
        # First extract ALL URLs
        raw_details = await evaluator.extract(
            prompt=prompt_extract_illness_details(illness_name),
            template_class=IllnessDetailsRaw,
            extraction_name=f"illness_details_raw_{illness_name}"
        )
        
        # Select two URLs from different sources
        url1, url2 = select_two_different_sources(raw_details.all_source_urls)
        
        # Create illness with selected URLs
        illness = Illness(
            name=illness_name,
            combined_symptoms=raw_details.combined_symptoms,
            source_url_1=url1,
            source_url_2=url2,
            vaccine_info=raw_details.vaccine_info
        )
        illnesses.append(illness)

    # Pad to ensure we have exactly 3 illnesses
    while len(illnesses) < 3:
        illnesses.append(Illness())

    # -------- 4. Build verification tree -------------------------------- #
    # Verify each illness
    for i, illness in enumerate(illnesses, 1):
        await verify_illness(evaluator, root, illness, i)

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()