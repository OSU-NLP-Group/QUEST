import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "phd_program_statistics"
TASK_DESCRIPTION = """
Please find the number of applications and the acceptance rate (percentage of applicants admitted, rounded to the nearest percent) for the year 2024 for the following PhD programs:

- UCSD Computer Science and Engineering PhD Program
- UMich Computer Science and Engineering PhD Program
- UCLA Computer Science PhD Program

For each program, clearly provide the link to the official page displaying the graduate admission statistics.
"""

EVAL_NOTES = """
First, compare the numbers with the ground truth. Next, check the source links: if a link matches the ground truth link, mark it as correct. Otherwise, use verify_by_urls to verify if the provided links substantiate the numbers.
"""

GROUND_TRUTH = {
    "UCSD": {
        "applications": "2147",
        "acceptance_rate": "8%",
        "ground_truth_url": "grad.ucsd.edu/about/grad-data/admissions.html"
    },
    "UMich": {
        "applications": "1301",
        "acceptance_rate": "6%",
        "ground_truth_url": "tableau.dsc.umich.edu/t/UM-Public/views/RackhamDoctoralProgramStatistics/ProgramStatistics"
    },
    "UCLA": {
        "applications": "1687",
        "acceptance_rate": "10%",
        "ground_truth_url": "grad.ucla.edu/graduate-program-statistics/admissions/?t=Annualsnapshot"
    }
}


class SingleProgramStats(BaseModel):
    """Statistics for a single PhD program"""
    applications: Optional[str] = Field(default=None, description="Number of applications")
    acceptance_rate: Optional[str] = Field(default=None, description="Acceptance rate percentage")
    urls: List[str] = Field(default_factory=list, description="Source URLs for the statistics")


def prompt_extract_single_program_stats(program_name: str) -> str:
    """Extraction prompt for a single PhD program's statistics"""
    return f"""
    Extract the PhD program admission statistics from the answer for the year 2024.

    Look for the following specific program:
    {program_name}

    Extract:
    - applications: The number of applications (as a string, exactly as written)
    - acceptance_rate: The acceptance rate percentage (as a string, exactly as written, including the % sign if present)
    - urls: ALL URLs mentioned in relation to this program's statistics (as a list)

    Extract information exactly as it appears in the text. If any field is not mentioned, set it to null.
    If the program is not mentioned at all, leave all fields as default (null/empty list).

    IMPORTANT: Only extract information specifically related to {program_name}. Do not include information from other programs.
    """


async def verify_program_stats(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        program_name: str,
        program_key: str,
        stats: SingleProgramStats,
        ground_truth: Dict[str, str]
) -> None:
    """
    Verify statistics for a single PhD program.

    Args:
        evaluator: The evaluator instance
        parent_node: Parent node to attach verifications to
        program_name: Full name of the program (e.g., "UCSD Computer Science and Engineering PhD Program")
        program_key: Key for ground truth lookup (e.g., "UCSD")
        stats: Extracted statistics for the program
        ground_truth: Ground truth data for the program
    """
    # Create a non-critical node for this program (to allow partial credit across programs)
    program_node = evaluator.add_parallel(
        id=f"{program_key.lower()}_program",
        desc=f"{program_name} statistics verification",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit across programs
    )

    # Check if program data exists
    has_data = bool(stats.applications or stats.acceptance_rate or stats.urls)

    data_exists_node = evaluator.add_custom_node(
        result=has_data,
        id=f"{program_key.lower()}_data_exists",
        desc=f"{program_name} data exists in answer",
        parent=program_node,
        critical=True  # Critical - if no data, skip all checks for this program
    )

    # Verify number of applications
    applications_node = evaluator.add_leaf(
        id=f"{program_key.lower()}_applications",
        desc=f"{program_name} - Number of applications matches ground truth ({ground_truth['applications']})",
        parent=program_node,
        critical=True  # Critical for this program's validity
    )

    await evaluator.verify(
        claim=f"The number ({stats.applications}) matches the ground truth value of {ground_truth['applications']}",
        node=applications_node,
        additional_instruction="Allow reasonable variations in number format (e.g., with or without commas). The numbers should be essentially the same value."
    )

    # Verify acceptance rate
    acceptance_rate_node = evaluator.add_leaf(
        id=f"{program_key.lower()}_acceptance_rate",
        desc=f"{program_name} - Acceptance rate matches ground truth ({ground_truth['acceptance_rate']})",
        parent=program_node,
        critical=True  # Critical for this program's validity
    )

    await evaluator.verify(
        claim=f"The acceptance rate extracted from the answer ({stats.acceptance_rate}) matches the ground truth value of {ground_truth['acceptance_rate']}",
        node=acceptance_rate_node,
        additional_instruction="Allow reasonable variations in percentage format (e.g., '8%' vs '8 percent' vs '8'). The percentage value should be the same."
    )

    # Verify source URL
    ground_truth_url = ground_truth['ground_truth_url']
    url_matches_gt = any(
        ground_truth_url.strip().rstrip('/').lower() in url.strip().rstrip('/').lower()
        for url in stats.urls
    )

    if url_matches_gt:
        # If URL matches ground truth, mark as correct directly
        url_node_1 = evaluator.add_custom_node(
            result=True,
            id=f"{program_key.lower()}_url_correct_applications",
            desc=f"{program_name} - Source URL matches ground truth for applications",
            parent=program_node,
            critical=True
        )

        url_node_2 = evaluator.add_custom_node(
            result=True,
            id=f"{program_key.lower()}_url_correct_acceptance",
            desc=f"{program_name} - Source URL matches ground truth for acceptance rate",
            parent=program_node,
            critical=True
        )
    else:
        # Otherwise, verify if any provided URL substantiates the statistics
        url_node_1 = evaluator.add_leaf(
            id=f"{program_key.lower()}_url_verification_applications",
            desc=f"{program_name} - Source URL substantiates the number of applications",
            parent=program_node,
            critical=True  # Critical - must have valid source
        )

        claim = f"For {program_name} in 2024, the number of applications is {stats.applications}"

        await evaluator.verify(
            claim=claim,
            node=url_node_1,
            sources=stats.urls,
            additional_instruction="Verify that the webpage contains the 2024 admission statistics for this specific PhD program. The numbers should match what is claimed."
        )

        url_node_2 = evaluator.add_leaf(
            id=f"{program_key.lower()}_url_verification_acceptance",
            desc=f"{program_name} - Source URL substantiates the acceptance rate",
            parent=program_node,
            critical=True  # Critical - must have valid source
        )

        claim = f"For {program_name} in 2024, the acceptance rate is {stats.acceptance_rate}"

        await evaluator.verify(
            claim=claim,
            node=url_node_2,
            sources=stats.urls,
            additional_instruction="Verify that the webpage contains the 2024 admission statistics for this specific PhD program. The numbers should match what is claimed."
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
    Main evaluation function for PhD program statistics task.

    Evaluates whether the answer correctly provides:
    1. Number of applications for each program
    2. Acceptance rate for each program
    3. Valid source URLs for the statistics
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Programs evaluated independently
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

    # -------- 2. Add ground truth info ---------------------------- #
    evaluator.add_ground_truth(GROUND_TRUTH, "phd_program_ground_truth")

    # -------- 3. Extract structured information separately for each program ---- #
    programs = [
        ("UCSD Computer Science and Engineering PhD Program", "UCSD"),
        ("UMich Computer Science and Engineering PhD Program", "UMich"),
        ("UCLA Computer Science PhD Program", "UCLA")
    ]

    extracted_stats = {}

    for program_name, program_key in programs:
        # Extract stats for each program individually
        stats = await evaluator.extract(
            prompt=prompt_extract_single_program_stats(program_name),
            template_class=SingleProgramStats,
            extraction_name=f"{program_key.lower()}_statistics_extraction",
        )
        extracted_stats[program_key] = stats

    # -------- 4. Build verification tree -------------------------- #

    # Verify each program's statistics
    for program_name, program_key in programs:
        await verify_program_stats(
            evaluator=evaluator,
            parent_node=root,
            program_name=program_name,
            program_key=program_key,
            stats=extracted_stats[program_key],
            ground_truth=GROUND_TRUTH[program_key]
        )

    # -------- 5. Return evaluation results ------------------------ #
    return evaluator.get_summary()