import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.llm_client.base_client import LLMClient

TASK_ID = "ios_num_iphone_model"
TASK_DESCRIPTION = """
Looking at the full timeline of iOS, for the 10 most recent iOS versions, please provide its initial release date, and how many iPhone models it supports. Note: According to the official classification, different model name suffixes are considered separate models — for example, versions with and without the "Plus" suffix are counted as two distinct iPhone models.
"""

GROUND_TRUTH = {
    "4": {"date": "June 21, 2010", "models": 3},
    "5": {"date": "October 12, 2011", "models": 3},
    "6": {"date": "September 19, 2012", "models": 4},
    "7": {"date": "September 18, 2013", "models": 5},
    "8": {"date": "September 17, 2014", "models": 6},
    "9": {"date": "September 16, 2015", "models": 9},
    "10": {"date": "September 13, 2016", "models": 10},
    "11": {"date": "September 19, 2017", "models": 11},
    "12": {"date": "September 17, 2018", "models": 14},
    "13": {"date": "September 19, 2019", "models": 15},
    "14": {"date": "September 16, 2020", "models": 19},
    "15": {"date": "September 24, 2021", "models": 24},
    "16": {"date": "September 12, 2022", "models": 23},
    "17": {"date": "September 22, 2023", "models": 24},
    "18": {"date": "September 20, 2024", "models": 29},
    "26": {"date": "September 15, 2025", "models": 30},
}

ALL_VERSION_KEYS = sorted(GROUND_TRUTH.keys(), key=lambda v: int(v))
LATEST_VERSION_KEYS = ALL_VERSION_KEYS[-10:]
LATEST_VERSION_INTS = [int(v) for v in LATEST_VERSION_KEYS]

class IOSVersionList(BaseModel):
    """List of iOS version numbers extracted from the answer"""
    versions: List[int] = Field(default_factory=list,
                                description="List of iOS version numbers as integers (e.g., [4, 5, 6])")


class IOSVersionDetails(BaseModel):
    """Details for a specific iOS version"""
    release_date: Optional[str] = Field(default=None, description="Release date of the iOS version")
    iphone_models_count: Optional[int] = Field(default=None, description="Number of iPhone models supported")
    urls: List[str] = Field(default_factory=list, description="URLs supporting this iOS version information")


def prompt_extract_ios_versions() -> str:
    """Extraction prompt for iOS version numbers"""
    return """
    Extract all the 10 most recent iOS version numbers mentioned in the answer.

    IMPORTANT:
    - Extract ONLY the version numbers as integers (e.g., if the answer mentions "iOS 4", extract 4)
    - If more than 10 versions are mentioned, keep the 10 highest version numbers
    - Only include versions that appear to be final answers (not just mentioned in passing)
    - If the answer provides a list or table of iOS versions, extract all version numbers from it
    - Return the version numbers as a list of integers
    """


def prompt_extract_version_details(version_num: int) -> str:
    """Extraction prompt for specific iOS version details"""
    return f"""
    Extract information specifically about iOS {version_num} from the answer.

    Look for:
    - release_date: The initial release date of iOS {version_num}. Extract exactly as written in the answer.
    - iphone_models_count: The number of iPhone models that iOS {version_num} supports. Extract as an integer.
    - urls: ALL URLs that are cited or referenced in relation to iOS {version_num}. Include any URLs that might support the release date or iPhone model count information for iOS {version_num}.

    IMPORTANT:
    - Only extract information that is specifically about iOS {version_num}
    - If the information is not found, return null for that field
    - Extract ALL relevant URLs, even if they appear elsewhere in the answer but relate to iOS {version_num}
    """


async def verify_ios_version(
        evaluator: Evaluator,
        parent_node,
        version_num: str,
        version_details: IOSVersionDetails,
        position_index: int
) -> None:
    """
    Verify a single iOS version's information.
    First check if it's in GT, then verify details if it is.
    """
    # Create a container node for this position
    version_node = evaluator.add_parallel(
        id=f"position_{position_index}_ios_{version_num}",
        desc=f"Position {position_index + 1}: iOS {version_num}",
        parent=parent_node,
        critical=False  # Non-critical to allow partial scoring
    )

    # Single critical check for GT presence
    evaluator.add_custom_node(
        result=(str(version_num) in GROUND_TRUTH),
        id=f"position_{position_index}_in_gt",
        desc=f"iOS {version_num} is one of the 10 most recent iOS versions",
        parent=version_node,
        critical=True  # Critical - if False, entire version_node scores 0
    )
    # Get the expected data for this version
    expected_data = GROUND_TRUTH.get(version_num, {"date": "unknown", "models": 0})

    # Check if all required information exists
    existence_node = evaluator.add_custom_node(
        result=(
                version_details.release_date is not None and
                version_details.iphone_models_count is not None and
                len(version_details.urls) > 0
        ),
        id=f"position_{position_index}_complete_info",
        desc=f"iOS {version_num} has complete information (date, model count, and URLs)",
        parent=version_node,
        critical=True
    )

    # Verify all information matches ground truth
    info_match_node = evaluator.add_leaf(
        id=f"position_{position_index}_info_match",
        desc=f"iOS {version_num} information matches ground truth",
        parent=version_node,
        critical=True
    )

    # Create string representations for comparison
    extracted_str = f"iOS {version_num}: (Release date: {version_details.release_date}), ({version_details.iphone_models_count} models)"
    expected_str = f"iOS {version_num}: (Release date: {expected_data['date']}), ({expected_data['models']} models)"

    await evaluator.verify(
        claim=f"The extracted information '{extracted_str}' matches the expected information '{expected_str}'",
        node=info_match_node,
        additional_instruction="Verify that the iOS version number, release date, and iPhone model count all match. Allow for reasonable or minor format variations (e.g., 'September 17, 2018' vs 'Sep 17, 2018')."
    )

    # Verify URL support for release date
    date_url_support_node = evaluator.add_leaf(
        id=f"position_{position_index}_date_url_support",
        desc=f"iOS {version_num} release date is supported by provided URLs",
        parent=version_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"iOS {version_num} was released on {version_details.release_date}",
        node=date_url_support_node,
        sources=version_details.urls,
        additional_instruction=f"Verify that the webpage confirms iOS {version_num} was released on or around {version_details.release_date}. The expected date is {expected_data['date']}."
    )


async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Main evaluation function for iOS version information task.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Add ground truth information
    evaluator.add_ground_truth({
        "expected_versions": [f"iOS {v}" for v in LATEST_VERSION_KEYS],
        "total_versions": len(LATEST_VERSION_KEYS)
    })

    # Step 1: Extract iOS version numbers
    extracted_versions = await evaluator.extract(
        prompt=prompt_extract_ios_versions(),
        template_class=IOSVersionList,
        extraction_name="ios_version_numbers"
    )

    extracted_version_ints = extracted_versions.versions
    extracted_version_set = set(extracted_version_ints)

    # Step 2: Extract details for each version and verify
    all_version_details = {}
    versions_found_in_gt = []
    versions_checked_labels: List[str] = []

    for position_idx, version_key in enumerate(LATEST_VERSION_KEYS):
        version_int = int(version_key)
        if version_int not in extracted_version_set:
            # Create placeholder node for missing version
            missing_node = evaluator.add_parallel(
                id=f"position_{position_idx}_missing_ios_{version_key}",
                desc=f"Position {position_idx + 1}: Missing iOS {version_key}",
                parent=root,
                critical=False
            )

            evaluator.add_custom_node(
                result=False,
                id=f"position_{position_idx}_ios_{version_key}_not_found",
                desc=f"iOS {version_key} not provided in the answer",
                parent=missing_node,
                critical=True
            )
            versions_checked_labels.append(f"Missing (iOS {version_key})")
        else:
            # Extract details for this version
            version_str = version_key
            details = await evaluator.extract(
                prompt=prompt_extract_version_details(version_int),
                template_class=IOSVersionDetails,
                extraction_name=f"ios_{version_str}_details"
            )
            all_version_details[version_str] = details

            # Track if this version is in GT
            if version_str in GROUND_TRUTH and f"iOS {version_str}" not in versions_found_in_gt:
                versions_found_in_gt.append(f"iOS {version_str}")

            # Verify this version
            await verify_ios_version(
                evaluator=evaluator,
                parent_node=root,
                version_num=version_str,
                version_details=details,
                position_index=position_idx
            )
            versions_checked_labels.append(f"iOS {version_str}")

    # Add custom info about extraction results
    evaluator.add_custom_info({
        "extracted_count": len(extracted_versions.versions),
        "checked_count": len(versions_found_in_gt),
        "expected_count": len(LATEST_VERSION_KEYS),
        "target_versions": [f"iOS {v}" for v in LATEST_VERSION_KEYS],
        "extracted_versions": [f"iOS {v}" for v in extracted_versions.versions],
        "versions_checked": versions_checked_labels,
        "valid_versions_found": versions_found_in_gt
    }, "extraction_summary")

    # Return evaluation summary
    return evaluator.get_summary()
