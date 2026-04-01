import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "artifact_repatriation"
TASK_DESCRIPTION = """
List 5 archaeological artifacts that have been repatriated from countries in Europe or North America to Asian countries since 2018. For each artifact, please provide the artifact name, country returning the artifact, country receiving the artifact (repatriated to), year of repatriation, and a verified news source (link).
"""

START_YEAR = 2018
CURRENT_YEAR = datetime.utcnow().year


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArtifactInfo(BaseModel):
    """Information about a single repatriated artifact."""
    name: Optional[str] = None
    country_returning: Optional[str] = None
    country_receiving: Optional[str] = None
    year: Optional[str] = None
    source_url: Optional[str] = None


class ArtifactNames(BaseModel):
    """Model for extracting just the names of artifacts."""
    names: List[str] = Field(default_factory=list)


class ArtifactDetails(BaseModel):
    """Model for extracting details about a specific artifact."""
    country_returning: Optional[str] = None
    country_receiving: Optional[str] = None
    year: Optional[str] = None
    news_source_url: Optional[str] = None


class ArtifactSources(BaseModel):
    """Model for extracting source URLs for a specific artifact."""
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_artifact_names() -> str:
    """Generate a prompt to extract the names of archaeological artifacts mentioned in the answer."""
    return """
    Extract the names of all archaeological artifacts mentioned in the answer that have been repatriated.
    Return a list of artifact names, even if there are more than 5.
    If an artifact is mentioned without a specific name, extract its description instead.
    """


def prompt_extract_artifact_details(artifact_name: str) -> str:
    """Generate a prompt to extract details about a specific artifact."""
    return f"""
    For the artifact named "{artifact_name}", extract the following information:

    1. country_returning: The country (in Europe or North America) that returned the artifact
    2. country_receiving: The Asian country that received the artifact
    3. year: The year of repatriation (on or after {START_YEAR})
    4. news_source_url: The URL specifically mentioned as a verified news source for this artifact

    If any information is missing, use null for that field.
    """


def prompt_extract_artifact_sources(artifact_name: str) -> str:
    """Generate a prompt to extract source URLs for a specific artifact."""
    return f"""
    Extract all URLs or web links mentioned in the answer that provide evidence or information about the repatriation of the artifact named "{artifact_name}".

    Focus on extracting full, complete URLs that could verify this artifact's repatriation details.
    """


# --------------------------------------------------------------------------- #
# Main verification function for individual artifacts                         #
# --------------------------------------------------------------------------- #
async def verify_artifact(
        evaluator: Evaluator,
        parent_node,
        artifact_name: str,
        artifact_details: ArtifactDetails,
        artifact_sources: ArtifactSources,
        artifact_idx: int,
) -> None:
    """
    Perform complete verification of a single artifact with two critical checks:
    1. Basic facts verification (year, countries)
    2. Source provenance verification
    """
    # Create a parallel node for this artifact (both checks must pass)
    artifact_node = evaluator.add_parallel(
        id=f"artifact_{artifact_idx}",
        desc=f"Artifact {artifact_idx + 1}: '{artifact_name}' is a valid archaeological artifact repatriated from Europe/North America to Asia since {START_YEAR} with verified sources",
        parent=parent_node,
        critical=False,  # Allow partial scoring across artifacts
    )

    required_info_present = (
        artifact_name is not None and artifact_name != "" and
        artifact_details.country_returning is not None and 
        artifact_details.country_receiving is not None and 
        artifact_details.year is not None and
        artifact_sources.source_urls
    )

    evaluator.add_custom_node(
        result=required_info_present,
        id=f"artifact_{artifact_idx}_complete",
        desc=f"Artifact '{artifact_name}' has all required information (countries, year, and source URLs)",
        parent=artifact_node,
        critical=True
    )

    # Critical Check 1: Basic facts verification
    facts_node = evaluator.add_leaf(
        id=f"artifact_{artifact_idx}_facts",
        desc=f"Artifact '{artifact_name}' has correct basic facts: year on or after {START_YEAR}, returning country in Europe/North America, receiving country in Asia",
        parent=artifact_node,
        critical=True,  # Must pass for this artifact to be valid
    )

    # Critical Check 2: Source provenance verification
    provenance_node = evaluator.add_leaf(
        id=f"artifact_{artifact_idx}_provenance",
        desc=f"All information about artifact '{artifact_name}' (name, countries, year) is substantiated by at least one provided source",
        parent=artifact_node,
        critical=True,  # Must pass for this artifact to be valid
    )

    # Verify basic facts with a single comprehensive claim
    facts_claim = f"""
    {artifact_details.year} is a year between {START_YEAR} and {CURRENT_YEAR} inclusive.
    '{artifact_details.country_returning}' is a country located in Europe or North America.
    '{artifact_details.country_receiving}' is a country located in Asia.
    """

    await evaluator.verify(
        claim=facts_claim,
        node=facts_node,
        additional_instruction="The three statements above must all be true to be valid.",
    )

    # Verify all information against sources with a comprehensive claim
    provenance_claim = f"""Carefully read the content of this webpage, and conduct the following check (We mainly want to check the information about '{artifact_name}' on this page. Really pay attention to the related information): 
    
    (1), this page discusses the repatriation of an archaeological artifact from '{artifact_name}'. For this point, plz broadly accept variants of the name of the artifact, as long as it is identifiable or inferrable as the same artifact. 
    
    (2), this artifact is repatriated from {artifact_details.country_returning} to {artifact_details.country_receiving} in {artifact_details.year}.
    """

    additional_instruction = """
    When verifying artifact names, be flexible with reasonable variations:
    - Different capitalization, punctuation, or formatting
    - The name also can be a reasonably correct summary or short name for the description in the webpage
    - Different word order or phrasing that describes the same object
    - Descriptive variations (e.g., "statue" vs "sculpture", numeric formats like "10th-Century" vs "10th-century")
    - The source may describe the artifact without using the exact name from the answer, as long as it's clearly identifiable as the same artifact

    Focus on whether the source substantiates the core facts about this artifact's repatriation, not on exact name matching.
    """

    await evaluator.verify(
        claim=provenance_claim,
        node=provenance_node,
        sources=artifact_sources.source_urls,
        additional_instruction=additional_instruction
    )

    # provenance_node_2 = evaluator.add_leaf(
    #     id_=f"artifact_{artifact_idx}_provenance",
    #     desc=f"All information about artifact '{artifact_name}' (name, countries, year) is substantiated by at least one provided source",
    #     parent=artifact_node,
    #     critical=True,  # Must pass for this artifact to be valid
    # )
    #
    # provenance_claim = f"""
    #     The page discusses the repatriation of an archaeological artifact from {artifact_details.country_returning} to {artifact_details.country_receiving} in {artifact_details.year}, and this artifact can be identified as '{artifact_name}' based on the description in the source.
    #     """
    #
    # additional_instruction = """
    #     When verifying artifact names, be flexible with reasonable variations:
    #     - Different capitalization, punctuation, or formatting
    #     - Different word order or phrasing that describes the same object
    #     - Descriptive variations (e.g., "statue" vs "sculpture", numeric formats like "10th-Century" vs "10th-century")
    #     - The source may describe the artifact without using the exact name from the answer, as long as it's clearly identifiable as the same artifact
    #
    #     The verification passes if:
    #     1. The source confirms an artifact matching the description was repatriated
    #     2. The countries and year match what's stated in the answer
    #     3. The artifact in the source can reasonably be identified as the one named in the answer, even if the exact wording differs
    #
    #     Focus on whether the source substantiates the core facts about this artifact's repatriation, not on exact name matching.
    #     """
    #
    # await evaluator.verify(
    #     claim=provenance_claim,
    #     node=provenance_node,
    #     sources=artifact_sources.source_urls,
    #     additional_instruction=additional_instruction
    # )


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
    Evaluate a single answer to the artifact repatriation task and return a structured result.
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

    # -------- 2. Build verification tree -------------------------------- #
    # Create a node to verify the artifacts (allows partial scoring across the 5 artifacts)
    artifacts_node = evaluator.add_parallel(
        id="artifacts",
        desc="Verification of up to 5 archaeological artifacts with complete repatriation details and verified sources",
        critical=False,
    )

    # -------- 3. Extract artifact names first ----------------------------- #
    extracted_names = await evaluator.extract(
        prompt=prompt_extract_artifact_names(),
        template_class=ArtifactNames,
        extraction_name="artifact_names"
    )
    # -------- 4. Pad the list to ensure we have exactly 5 items ----------- #
    artifact_names_padded = extracted_names.names[:5]  # Take first 5 if more
    while len(artifact_names_padded) < 5:
        artifact_names_padded.append(None)  # Pad with None

    # -------- 5. Extract details and sources for each artifact ------------ #
    artifact_info_list = []

    for i in range(5):
        artifact_name = artifact_names_padded[i]
        
        if artifact_name:
            # Extract details for real artifacts
            details = await evaluator.extract(
                prompt=prompt_extract_artifact_details(artifact_name),
                template_class=ArtifactDetails,
                extraction_name=f"artifact_{i}_details"
            )
            
            # Extract sources for real artifacts
            sources = await evaluator.extract(
                prompt=prompt_extract_artifact_sources(artifact_name),
                template_class=ArtifactSources,
                extraction_name=f"artifact_{i}_sources"
            )
            
            # Create artifact info for the list
            artifact_info = ArtifactInfo(
                name=artifact_name,
                country_returning=details.country_returning,
                country_receiving=details.country_receiving,
                year=details.year,
                source_url=details.news_source_url
            )
            artifact_info_list.append(artifact_info)
        else:
            # Create empty models for missing artifacts
            details = ArtifactDetails()
            sources = ArtifactSources()
        
        # Verify this artifact (works for both real and empty artifacts)
        await verify_artifact(
            evaluator=evaluator,
            parent_node=artifacts_node,
            artifact_name=artifact_name,
            artifact_details=details,
            artifact_sources=sources,
            artifact_idx=i,
        )

    # -------- 6. Add custom info for artifacts ----------------------------- #
    evaluator.add_custom_info({
        "artifacts": [a.dict() for a in artifact_info_list],
        "artifact_names": extracted_names.names[:5],
        "total_artifacts_found": len(extracted_names.names),
        "artifacts_verified": len(artifact_info_list)
    }, "artifact_summary")

    # -------- 7. Aggregate score and return structured result ------------- #
    # Calculate final score by triggering aggregation
    evaluator.root.compute_score(mutate=True)
    
    return evaluator.get_summary()
