import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hf_vqa_dataset"
TASK_DESCRIPTION = """
I'm looking into large-scale English visual question answering datasets hosted on HuggingFace, particularly those released by academic institutions in the US.

Please find two English visual question answering datasets with more than 1M rows and provided with Croissant metadata, released by a research group from US universities with their HuggingFace profile explicitly stating the university affiliation. For each dataset, show me the link to the dataset on HuggingFace and a link to the HuggingFace profile of the organization.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class DatasetInfo(BaseModel):
    dataset_url: Optional[str] = None
    organization_url: Optional[str] = None


class ExtractedInfo(BaseModel):
    datasets: List[DatasetInfo] = Field(default_factory=list)


# Prompt for extracting dataset information from the answer
def prompt_extract_datasets() -> str:
    return """
    Extract information about the datasets mentioned in the answer. 
    For each dataset, extract:
    1. The URL to the dataset on HuggingFace (should start with https://huggingface.co/datasets/)
    2. The URL to the organization's profile on HuggingFace (should start with https://huggingface.co/ and point to an organization profile)

    Use the format defined in the ExtractedInfo model, with a list of DatasetInfo objects.
    Return null for any fields that are not explicitly mentioned in the answer.
    Only extract valid HuggingFace URLs. Do not extract dataset names as they are not required for verification.
    """


# --------------------------------------------------------------------------- #
# Dataset verification functions                                              #
# --------------------------------------------------------------------------- #
async def verify_dataset(
        evaluator: Evaluator,
        parent_node,
        dataset_info: DatasetInfo,
        dataset_index: int,
) -> None:
    """
    Verify a dataset against all required criteria using sequential evaluation.
    """
    # 1. Combined existence check for both URLs
    has_dataset_url = (
        dataset_info.dataset_url is not None and
        dataset_info.dataset_url.strip() != "" and
        "huggingface" in dataset_info.dataset_url and 
        "datasets" in dataset_info.dataset_url
    )
    has_org_url = (
        dataset_info.organization_url is not None and
        dataset_info.organization_url.strip() != "" and 
        "huggingface" in dataset_info.organization_url
    )
    
    links_exist_node = evaluator.add_custom_node(
        result=has_dataset_url and has_org_url,
        id=f"dataset_{dataset_index}_links_exist",
        desc="Check if both dataset URL and organization URL are provided as valid HuggingFace links",
        parent=parent_node,
        critical=True
    )

    # 2. Verify organization links match
    org_matches_node = evaluator.add_leaf(
        id=f"dataset_{dataset_index}_org_matches",
        desc="The organization URL corresponds to the same organization that published the dataset",
        parent=parent_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The dataset URL '{dataset_info.dataset_url}' and organization URL '{dataset_info.organization_url}' both point to the same organization or organizations at the same university on HuggingFace. For example, if the dataset is from 'osunlp' organization, both URLs may clearly indicate this same organization (like https://huggingface.co/datasets/osunlp/dataset-name and https://huggingface.co/osunlp), or they should both clearly indicate the same academic institution (like 'osunlp' and 'ohiostate' both referring to OSU).",
        node=org_matches_node,
        additional_instruction="Analyze the URL structure and organization names to determine if they refer to the same entity or organizations from the same university. HuggingFace dataset URLs typically follow the pattern https://huggingface.co/datasets/[organization]/[dataset-name], and organization profiles are at https://huggingface.co/[organization]. Check if the organization names are identical or clearly refer to the same institution."
    )

    # 3. Verify dataset properties
    dataset_props_node = evaluator.add_leaf(
        id=f"dataset_{dataset_index}_properties",
        desc="The dataset is a VQA dataset in English with 1M+ rows and Croissant metadata",
        parent=parent_node,
        critical=True,
    )

    await evaluator.verify(
        claim="This is a Huggingface Dataset page. And, the dataset is a visual question answering (VQA) dataset in English with more than 1 million rows/examples and has Croissant metadata",
        node=dataset_props_node,
        sources=dataset_info.dataset_url,
        additional_instruction="Check the HuggingFace dataset page for the following criteria: 1) Task tags should include 'visual-question-answering' or similar VQA-related tags, 2) Language tags should include 'en' or 'English' or something similar 3) Size tags should indicate 1M-10M, 10M-100M, or similar large scale (>1M examples), or, there can be the exact or approximate numbers about the number of rows, check whether that is more than 1M 4) Library tags should include 'croissant' or similar metadata indicators. Focus primarily on the official tags and metadata shown on the HuggingFace page. If these tags are not explicitly available, look for clear statements in the dataset description that indicate these properties. For example, the dataset description may mention 'visual question answering', 'English', 'xxx rows', and 'Croissant metadata'."
    )

    # 4. Verify US university affiliation
    us_univ_node = evaluator.add_leaf(
        id=f"dataset_{dataset_index}_us_university",
        desc="The organization's HuggingFace profile explicitly states they are a research group from a US university",
        parent=parent_node,
        critical=True,
    )

    await evaluator.verify(
        claim="This is a webpage of a Huggingface Organization. And, the name or the profile explicitly states or indicates they are a university, or a research group from a US university, or similar",
        node=us_univ_node,
        sources=dataset_info.organization_url,
        additional_instruction="Check if the organization's HuggingFace profile page explicitly mentions affiliation with a university in the United States or a sub-institution under a university in the US. Look for clear statements in the profile description, bio, or organization name that indicate they are a research group from a US university (for example, 'OSU NLP Group' for The Ohio State University NLP group, or explicit mentions like 'University of X' where X is a US state/institution)."
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
    Evaluate an answer to the HuggingFace VQA dataset task.

    The evaluation checks if two VQA datasets are provided that meet all criteria:
    1. Both dataset and organization links are provided as valid HuggingFace URLs
    2. The organization link corresponds to the same organization that published the dataset
    3. The dataset meets all property requirements (VQA, English, 1M+ rows, Croissant metadata)
    4. The organization's profile explicitly states they are a research group from a US university
    """
    # Set up evaluator
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

    # Extract dataset information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_datasets(),
        template_class=ExtractedInfo,
        extraction_name="extracted_datasets"
    )

    # Pad datasets to ensure we have exactly 2
    datasets = list(extracted_info.datasets)
    while len(datasets) < 2:
        datasets.append(DatasetInfo())

    # Verify each dataset
    for i, dataset_info in enumerate(datasets[:2]):  # Only process first 2
        # Direct sequential parent for all dataset verifications
        dataset_node = evaluator.add_sequential(
            id=f"dataset_{i + 1}",
            desc=f"Dataset {i + 1}: Meets all task requirements",
            parent=root,
            critical=False,  # Non-critical to allow partial credit
        )
        
        await verify_dataset(evaluator, dataset_node, dataset_info, i + 1)

    # Return structured result
    return evaluator.get_summary()