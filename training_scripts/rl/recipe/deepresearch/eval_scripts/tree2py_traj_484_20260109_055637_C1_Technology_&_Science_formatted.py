import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "google_willow_info"
TASK_DESCRIPTION = (
    "On December 9, 2024, Google announced a new quantum chip called Willow. "
    "Please provide the following information about this quantum chip: "
    "(1) the official announcement date, (2) the total number of qubits in the chip, and "
    "(3) the performance comparison on the random circuit sampling (RCS) benchmark between Willow and today's "
    "fastest classical supercomputers, including how long Willow took to complete the computation and how long a "
    "classical supercomputer would take."
)

EXPECTED_CHIP_NAME = "Willow"
EXPECTED_ANNOUNCEMENT_DATE = "December 9, 2024"
EXPECTED_QUBITS = "105"
EXPECTED_RCS_BENCHMARK = "random circuit sampling (RCS)"
EXPECTED_WILLOW_RCS_RUNTIME = "under five minutes"
EXPECTED_CLASSICAL_RUNTIME = "approximately 10^25 years"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WillowInfo(BaseModel):
    chip_name: Optional[str] = None
    announcement_date: Optional[str] = None
    qubit_count: Optional[str] = None
    rcs_benchmark_name: Optional[str] = None
    willow_rcs_runtime: Optional[str] = None
    classical_rcs_runtime: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_willow_info() -> str:
    return (
        "Extract the specific facts the answer provides about Google's quantum chip Willow. "
        "Return a JSON object with the following fields:\n"
        "1) chip_name: The chip name explicitly stated in the answer (string). If not stated, null.\n"
        "2) announcement_date: The date the answer claims as the official announcement date (string, keep the format as written, e.g., 'December 9, 2024', 'Dec 9, 2024'). If not stated, null.\n"
        "3) qubit_count: The total number of qubits as stated in the answer (string; do not coerce to a number; e.g., '105', '105 qubits', '105‑qubit'). If not stated, null.\n"
        "4) rcs_benchmark_name: The benchmark name used in the comparison (string; e.g., 'random circuit sampling', 'RCS'). If not stated, null.\n"
        "5) willow_rcs_runtime: The runtime the answer claims Willow achieved on the RCS benchmark (string; e.g., 'under five minutes', '<5 minutes', '4 minutes'). If not stated, null.\n"
        "6) classical_rcs_runtime: The runtime the answer claims for a classical supercomputer on the same RCS benchmark (string; e.g., 'approximately 10^25 years', '10 septillion years'). If not stated, null.\n"
        "7) official_urls: An array of all URLs in the answer that appear to be citations for Google's official announcement about Willow. "
        "Only include URLs explicitly present in the answer. This can include domains such as blog.google, ai.google, research.google, quantumai.google. "
        "If the answer mentions an official announcement without a URL, return an empty array.\n"
        "Do not invent or infer any information not explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_willow_info(
    evaluator: Evaluator,
    extracted: WillowInfo,
) -> None:
    """
    Build the verification tree according to the rubric and run checks.
    """
    # Add the top-level critical parallel node for Google's Willow information
    root_group = evaluator.add_parallel(
        id="Google_Willow_Information",
        desc="Provide required, constraint-consistent information about Google's quantum chip Willow, with sourcing from Google's official announcement.",
        parent=evaluator.root,
        critical=True,
    )

    # Helper: pick sources (None -> simple_verify; Non-empty -> verify_by_urls)
    sources_for_verification = extracted.official_urls if extracted.official_urls else None

    # Chip Name leaf (critical)
    chip_name_leaf = evaluator.add_leaf(
        id="Chip_Name",
        desc='Correctly identify the quantum chip name as "Willow".',
        parent=root_group,
        critical=True,
    )
    claim_chip_name = "The quantum chip's name is 'Willow'."
    await evaluator.verify(
        claim=claim_chip_name,
        node=chip_name_leaf,
        sources=sources_for_verification,
        additional_instruction=(
            "Also check that the answer explicitly names the chip as 'Willow'. "
            "Treat minor casing or formatting variants as equivalent (e.g., 'willow', 'Willow chip')."
        ),
    )

    # Announcement Date leaf (critical)
    announcement_leaf = evaluator.add_leaf(
        id="Announcement_Date",
        desc="State the official announcement date as December 9, 2024.",
        parent=root_group,
        critical=True,
    )
    claim_announcement = "The official announcement date for the Willow quantum chip is December 9, 2024."
    await evaluator.verify(
        claim=claim_announcement,
        node=announcement_leaf,
        sources=sources_for_verification,
        additional_instruction=(
            "Also check that the answer states the same date. Accept minor format variants such as 'Dec 9, 2024' or 'December 9, 2024'."
        ),
    )

    # Number of Qubits leaf (critical)
    qubits_leaf = evaluator.add_leaf(
        id="Number_of_Qubits",
        desc="State the total number of qubits as 105.",
        parent=root_group,
        critical=True,
    )
    claim_qubits = "The Willow quantum chip has a total of 105 qubits."
    await evaluator.verify(
        claim=claim_qubits,
        node=qubits_leaf,
        sources=sources_for_verification,
        additional_instruction=(
            "Also check that the answer states '105' (or equivalent phrasing such as '105 qubits', '105‑qubit')."
        ),
    )

    # Benchmark Performance (RCS) group (critical parallel)
    rcs_group = evaluator.add_parallel(
        id="Benchmark_Performance_RCS",
        desc="Provide the required RCS benchmark performance comparison information.",
        parent=root_group,
        critical=True,
    )

    # Mentions RCS Benchmark leaf (critical)
    rcs_benchmark_leaf = evaluator.add_leaf(
        id="Mentions_RCS_Benchmark",
        desc="Explicitly identify the benchmark as the random circuit sampling (RCS) benchmark.",
        parent=rcs_group,
        critical=True,
    )
    claim_rcs_benchmark = "The performance benchmark used for Willow is random circuit sampling (RCS)."
    await evaluator.verify(
        claim=claim_rcs_benchmark,
        node=rcs_benchmark_leaf,
        sources=sources_for_verification,
        additional_instruction=(
            "Also ensure the answer explicitly identifies 'random circuit sampling' or 'RCS' as the benchmark."
        ),
    )

    # Willow RCS Runtime leaf (critical)
    willow_runtime_leaf = evaluator.add_leaf(
        id="Willow_RCS_Runtime",
        desc="State that Willow completed the RCS computation in under five minutes.",
        parent=rcs_group,
        critical=True,
    )
    claim_willow_runtime = "For the RCS benchmark, Willow completed the computation in under five minutes."
    await evaluator.verify(
        claim=claim_willow_runtime,
        node=willow_runtime_leaf,
        sources=sources_for_verification,
        additional_instruction=(
            "Also ensure the answer conveys 'under five minutes' or an equivalent phrasing such as '< 5 minutes', "
            "'less than five minutes', or 'sub-five-minute'."
        ),
    )

    # Classical Runtime Comparison leaf (critical)
    classical_runtime_leaf = evaluator.add_leaf(
        id="Classical_Runtime_Comparison",
        desc="State that the same RCS computation would take approximately 10^25 years (10 septillion years) on one of today's fastest classical supercomputers.",
        parent=rcs_group,
        critical=True,
    )
    claim_classical_runtime = (
        "The same RCS computation would take approximately 10^25 years (10 septillion years) on one of today's fastest classical supercomputers."
    )
    await evaluator.verify(
        claim=claim_classical_runtime,
        node=classical_runtime_leaf,
        sources=sources_for_verification,
        additional_instruction=(
            "Also ensure the answer states an equivalent phrasing such as 'approximately 10^25 years', "
            "'~10^25 years', '10 septillion years', or 'ten septillion years'."
        ),
    )

    # Sourcing from Official Announcement leaf (critical)
    sourcing_leaf = evaluator.add_leaf(
        id="Sourcing_From_Official_Announcement",
        desc="Provide a citation/reference indicating the information is sourced from Google's official announcement about Willow.",
        parent=root_group,
        critical=True,
    )
    claim_official_source = (
        "At least one of the provided URLs is Google's official announcement about the Willow quantum chip."
    )
    await evaluator.verify(
        claim=claim_official_source,
        node=sourcing_leaf,
        sources=extracted.official_urls,  # Must use verify_by_urls; if empty, it will fail
        additional_instruction=(
            "Treat as official Google sources pages under domains such as blog.google, ai.google, research.google, or quantumai.google. "
            "The page should clearly be an official announcement or blog/news post from Google about the Willow chip."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Google Willow information task and return a structured result.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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
        default_model=model,
    )

    # Extract structured Willow information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_willow_info(),
        template_class=WillowInfo,
        extraction_name="willow_info",
    )

    # Add ground truth information for transparency
    evaluator.add_ground_truth(
        {
            "expected_chip_name": EXPECTED_CHIP_NAME,
            "expected_announcement_date": EXPECTED_ANNOUNCEMENT_DATE,
            "expected_qubits": EXPECTED_QUBITS,
            "expected_benchmark": EXPECTED_RCS_BENCHMARK,
            "expected_willow_runtime": EXPECTED_WILLOW_RCS_RUNTIME,
            "expected_classical_runtime": EXPECTED_CLASSICAL_RUNTIME,
            "note": "Verification relies on Google's official announcement as the primary source.",
        },
        gt_type="expected_values",
    )

    # Build verification tree and run checks
    await build_and_verify_willow_info(evaluator, extracted)

    # Return evaluator summary
    return evaluator.get_summary()