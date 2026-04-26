import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "quantinuum_helios_physical_qubits_nov2025"
TASK_DESCRIPTION = "How many physical qubits does Quantinuum's Helios quantum computer, announced in November 2025, have?"


# --------------------------------------------------------------------------- #
# Data Models for Extraction                                                  #
# --------------------------------------------------------------------------- #
class HeliosAnswerExtraction(BaseModel):
    system_name: Optional[str] = None
    company: Optional[str] = None
    announcement_time: Optional[str] = None  # e.g., "November 2025" or "Nov 2025"
    physical_qubit_count: Optional[str] = None  # keep as string to allow variants like "50,000"
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_helios_answer() -> str:
    return """
    Extract the key details the answer provides about Quantinuum's Helios quantum computer.
    Return a JSON object with the following fields:
    - system_name: The name of the system as stated in the answer (e.g., "Helios", "Quantinuum Helios").
    - company: The company or organization associated with the system (e.g., "Quantinuum").
    - announcement_time: The announcement timing as stated (e.g., "November 2025", "Nov 2025").
    - physical_qubit_count: The number of physical qubits as explicitly stated in the answer (verbatim, including commas or units like 'k' if used). This should refer to PHYSICAL (hardware) qubits, not logical qubits.
    - source_urls: All URLs cited in the answer that are offered as references. Include every URL you find in the answer, whether inline, in markdown, or in a sources section.
    If any field is missing in the answer, set it to null (or [] for source_urls).
    """


# --------------------------------------------------------------------------- #
# Verification Logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: HeliosAnswerExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create the main critical parallel node per rubric
    main_node = evaluator.add_parallel(
        id="Helios_Physical_Qubit_Specification",
        desc="Answer provides the physical qubit count of Quantinuum's Helios quantum computer announced in November 2025, with proper reference",
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: System Identification
    # Verify that the answer itself identifies the system as Quantinuum's Helios and mentions the November 2025 announcement.
    sys_id_node = evaluator.add_leaf(
        id="System_Identification",
        desc="Answer correctly identifies the quantum computer as Quantinuum's Helios system announced in November 2025",
        parent=main_node,
        critical=True
    )
    sys_id_claim = (
        "Within the answer text, the referenced system is Quantinuum's Helios, and the answer also indicates that it was announced in November 2025."
    )
    await evaluator.verify(
        claim=sys_id_claim,
        node=sys_id_node,
        additional_instruction=(
            "Judge only based on the answer text. Accept reasonable variants: 'Quantinuum Helios', "
            "'Helios by Quantinuum', and month formats like 'Nov 2025' or specific November 2025 dates. "
            "Both identity (Quantinuum + Helios) and the November 2025 announcement timing must be present."
        ),
    )

    # Leaf 2: Physical Qubit Count
    # If the answer extracted a concrete number, verify that number against the cited sources when available.
    # Otherwise, fall back to checking the answer states a specific physical qubit number.
    qubit_leaf = evaluator.add_leaf(
        id="Physical_Qubit_Count",
        desc="Answer states the number of physical qubits in the Helios system",
        parent=main_node,
        critical=True
    )

    if extraction.physical_qubit_count and extraction.physical_qubit_count.strip():
        qubit_claim = (
            f"Quantinuum's Helios quantum computer has {extraction.physical_qubit_count.strip()} physical qubits."
        )
        await evaluator.verify(
            claim=qubit_claim,
            node=qubit_leaf,
            sources=extraction.source_urls if extraction.source_urls else None,
            additional_instruction=(
                "Use the provided sources (if any) to determine whether the stated count refers to physical (hardware) qubits. "
                "If multiple numbers appear, choose the one explicitly labeled as 'physical' or clearly referring to total hardware qubits. "
                "Allow minor formatting differences (commas, spaces, 'k' for thousand)."
            ),
        )
    else:
        # No concrete number extracted; verify at least that the answer asserts a specific physical qubit count.
        fallback_claim = (
            "The answer explicitly states a specific number for the physical (hardware) qubits of Quantinuum's Helios system."
        )
        await evaluator.verify(
            claim=fallback_claim,
            node=qubit_leaf,
            additional_instruction=(
                "Check that the answer contains a concrete numeric value indicating physical qubits (not logical qubits). "
                "The presence of a specific number is required."
            ),
        )

    # Leaf 3: Official Source Reference
    # Check that at least one cited URL is an official Quantinuum announcement or an equivalent authoritative source
    # about the Helios system (the November 2025 announcement).
    official_ref_node = evaluator.add_leaf(
        id="Official_Source_Reference",
        desc="Answer references the official Quantinuum announcement or equivalent authoritative source",
        parent=main_node,
        critical=True
    )
    official_claim = (
        "This page is an official announcement from Quantinuum, or an equivalently authoritative source (e.g., a company press release/newsroom or a recognized wire service), about the Helios quantum computer announced in November 2025."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_ref_node,
        sources=extraction.source_urls if extraction.source_urls else None,
        additional_instruction=(
            "Accept official pages on quantinuum.com (e.g., press releases/newsroom) or equivalent authoritative outlets "
            "such as Business Wire, PR Newswire, or clearly official corporate blogs. The page should explicitly discuss "
            "Quantinuum's Helios and relate to the November 2025 announcement. If no URLs are provided, this should fail."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Quantinuum Helios physical qubit count task.
    """
    # Initialize evaluator (root is always non-critical by framework design)
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_helios_answer(),
        template_class=HeliosAnswerExtraction,
        extraction_name="helios_answer_extraction",
    )

    # Optional: add ground truth/context info slot (no known GT number here)
    evaluator.add_ground_truth({
        "expected_system": "Quantinuum Helios",
        "expected_announcement_time": "November 2025 (as per task)",
        "note": "This task evaluates whether the answer states the physical qubit count and cites an official/authoritative source."
    }, gt_type="task_context")

    # Build the verification tree and run checks
    await build_and_verify_tree(evaluator, extraction)

    # Return structured summary
    return evaluator.get_summary()