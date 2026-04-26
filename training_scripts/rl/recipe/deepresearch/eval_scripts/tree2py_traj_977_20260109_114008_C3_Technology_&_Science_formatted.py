import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "quantum_verifiable_advantage_2025"
TASK_DESCRIPTION = """
In October 2025, a significant quantum computing breakthrough was published in the peer-reviewed journal Nature, representing what the authors described as the first demonstration of "verifiable quantum advantage" on hardware. This achievement utilized an algorithm based on Out-of-Time-Order Correlators (OTOC) to perform computations that could be verified by replicating the results on another quantum computer of similar caliber.

Identify this specific quantum computing achievement and provide the following verified information:

1. The exact title of the Nature publication
2. The specific name given to the algorithm implementation
3. The number of qubits used in the quantum processor
4. The claimed speedup factor compared to classical supercomputers
5. The primary institution or organization responsible for the development, along with a reference URL that verifies these details
""".strip()

EXPECTED = {
    "journal": "Nature",
    "publication_month_year": "October 2025",
    "publication_title": "Observation of constructive interference at the edge of quantum ergodicity",
    "algorithm_approach": "OTOC",
    "algorithm_impl_name": "Quantum Echoes",
    "architecture": "superconducting",
    "qubit_count": "105",
    "speedup": "approximately 13,000×",
    "primary_institution": "Google Quantum AI"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QuantumAchievementExtraction(BaseModel):
    publication_title: Optional[str] = None
    journal: Optional[str] = None
    publication_month_year: Optional[str] = None
    algorithm_approach: Optional[str] = None
    algorithm_impl_name: Optional[str] = None
    architecture: Optional[str] = None
    qubit_count: Optional[str] = None
    speedup_factor: Optional[str] = None
    primary_institution: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_quantum_achievement() -> str:
    return """
    Extract the following fields exactly as they are explicitly provided in the answer text. Do not invent or normalize; preserve the wording and formatting used by the answer.

    Fields to extract:
    - publication_title: The exact title of the Nature publication.
    - journal: The named journal (e.g., "Nature").
    - publication_month_year: The publication month and year string (e.g., "October 2025"). If only a date is given, reduce to month and year if unambiguous; otherwise keep the provided date string.
    - algorithm_approach: The core computational method (e.g., mention of "OTOC" or "Out-of-Time-Order Correlator").
    - algorithm_impl_name: The specific algorithm implementation name (e.g., "Quantum Echoes").
    - architecture: The hardware architecture used (e.g., "superconducting", "superconducting quantum processor").
    - qubit_count: The stated number of qubits, as a string (examples: "105", "105-qubit", "105 qubits").
    - speedup_factor: The claimed speedup vs classical supercomputers, as a string (examples: "~13,000×", "about 13,000x", "approximately 13k times").
    - primary_institution: The primary responsible institution/organization (e.g., "Google Quantum AI").
    - reference_urls: An array of all URLs explicitly included in the answer that are meant to support or verify these details (Nature article page, press releases, blogs, preprints, etc.). Only include valid URLs.

    If any field is not present in the answer, return null for single-value fields and an empty array for 'reference_urls'.
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def add_publication_details_checks(evaluator: Evaluator, parent_node) -> None:
    """
    Add publication detail verifications (all critical under a critical parent).
    """
    pub_node = evaluator.add_parallel(
        id="Publication_Details",
        desc="Publication identification details satisfy the constraints.",
        parent=parent_node,
        critical=True
    )

    # Journal is Nature
    journal_node = evaluator.add_leaf(
        id="Journal_Is_Nature",
        desc="States/identifies that the publication is in the peer-reviewed journal Nature.",
        parent=pub_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly identifies the publication as appearing in the peer-reviewed journal 'Nature'.",
        node=journal_node,
        additional_instruction="Check the answer text only. Accept 'Nature' or 'Nature (journal)' as equivalent."
    )

    # Publication date is October 2025
    date_node = evaluator.add_leaf(
        id="Publication_Date_Is_October_2025",
        desc="States/identifies that the publication occurred in October 2025.",
        parent=pub_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly indicates the publication occurred in October 2025.",
        node=date_node,
        additional_instruction="Check the answer text only. Accept minor formatting variants such as 'Oct. 2025'."
    )

    # Exact publication title match
    title_node = evaluator.add_leaf(
        id="Publication_Title_Matches_Constraint",
        desc="Provides the exact publication title and it matches the constrained title: 'Observation of constructive interference at the edge of quantum ergodicity'.",
        parent=pub_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer provides the exact publication title: 'Observation of constructive interference at the edge of quantum ergodicity'.",
        node=title_node,
        additional_instruction="Be strict: the provided title must match exactly in wording. Minor differences in capitalization or hyphen/emdash usage can be tolerated only if the words and order are identical."
    )


async def add_algorithm_details_checks(evaluator: Evaluator, parent_node) -> None:
    """
    Add algorithm detail verifications (all critical).
    """
    algo_node = evaluator.add_parallel(
        id="Algorithm_Details",
        desc="Algorithm details satisfy the constraints.",
        parent=parent_node,
        critical=True
    )

    # OTOC-based algorithm
    otoc_node = evaluator.add_leaf(
        id="Algorithm_Is_OTOC_Based",
        desc="States/identifies that the core computational method uses an Out-of-Time-Order Correlator (OTOC) algorithm.",
        parent=algo_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the core computational method uses an Out-of-Time-Order Correlator (OTOC) algorithm.",
        node=otoc_node,
        additional_instruction="Check the answer text only. Accept 'OTOC' or 'Out-of-Time-Order Correlator' as sufficient evidence."
    )

    # Algorithm implementation name: Quantum Echoes
    impl_node = evaluator.add_leaf(
        id="Algorithm_Implementation_Name_Matches_Constraint",
        desc="Provides the specific algorithm implementation name and it matches the constrained name: 'Quantum Echoes'.",
        parent=algo_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly names the algorithm implementation as 'Quantum Echoes'.",
        node=impl_node,
        additional_instruction="Check the answer text only. The provided name must be 'Quantum Echoes'."
    )


async def add_hardware_details_checks(evaluator: Evaluator, parent_node) -> None:
    """
    Add hardware detail verifications (all critical).
    """
    hw_node = evaluator.add_parallel(
        id="Hardware_Details",
        desc="Hardware details satisfy the constraints.",
        parent=parent_node,
        critical=True
    )

    # Architecture superconducting
    arch_node = evaluator.add_leaf(
        id="Architecture_Is_Superconducting",
        desc="States/identifies that the implementation is on a superconducting quantum processor architecture.",
        parent=hw_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the implementation is on a superconducting quantum processor.",
        node=arch_node,
        additional_instruction="Check the answer text only. Accept 'superconducting', 'superconducting qubits', or 'superconducting transmon' as sufficient."
    )

    # Qubit count exactly 105
    qubits_node = evaluator.add_leaf(
        id="Qubit_Count_Exactly_105",
        desc="States that the quantum processor uses exactly 105 qubits.",
        parent=hw_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the quantum processor uses exactly 105 qubits.",
        node=qubits_node,
        additional_instruction="Check the answer text only. Accept forms like '105-qubit', '105 qubits', or 'a 105-qubit processor'."
    )


async def add_performance_checks(evaluator: Evaluator, parent_node) -> None:
    """
    Add performance/advantage verifications (all critical).
    """
    perf_node = evaluator.add_parallel(
        id="Performance_And_Advantage",
        desc="Performance/advantage claims satisfy the constraints.",
        parent=parent_node,
        critical=True
    )

    # Speedup approximately 13,000x
    speed_node = evaluator.add_leaf(
        id="Speedup_Approximately_13000x",
        desc="States a claimed speedup of approximately 13,000× compared to classical supercomputers.",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states an approximate speedup of about 13,000× compared to classical supercomputers.",
        node=speed_node,
        additional_instruction="Check the answer text only. Accept notation variants like '13,000x', '~13k×', 'roughly 13,000 times'."
    )

    # Verifiable quantum advantage definition met
    vqa_node = evaluator.add_leaf(
        id="Verifiable_Quantum_Advantage_Definition_Met",
        desc="States/reflects that the result is 'verifiable quantum advantage' in the sense that it can be replicated/verified on another quantum computer of similar caliber.",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the result constitutes 'verifiable quantum advantage' because it can be verified by reproducing the computation on another quantum computer of similar caliber.",
        node=vqa_node,
        additional_instruction="Check the answer text only. The notion of reproducibility on another comparable quantum device must be present."
    )


async def add_institution_and_references_checks(
    evaluator: Evaluator,
    parent_node,
    extraction: QuantumAchievementExtraction
) -> None:
    """
    Add institution and references verifications (all critical).
    """
    inst_node = evaluator.add_parallel(
        id="Institution_And_References",
        desc="Institution attribution and citation requirements are met.",
        parent=parent_node,
        critical=True
    )

    # Primary institution: Google Quantum AI
    inst_leaf = evaluator.add_leaf(
        id="Primary_Institution_Matches_Constraint",
        desc="Identifies the primary responsible institution/organization and it matches the constraint: Google Quantum AI (alone or in collaboration with partners).",
        parent=inst_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer identifies the primary responsible organization as 'Google Quantum AI'.",
        node=inst_leaf,
        additional_instruction="Check the answer text only. Accept variants like 'Google Quantum AI team' or 'Google Research (Quantum AI)'."
    )

    # Provides at least one reference URL
    urls = extraction.reference_urls or []
    has_urls = any(isinstance(u, str) and u.strip() for u in urls)
    evaluator.add_custom_node(
        result=has_urls,
        id="Provides_Reference_URL",
        desc="Provides at least one reference URL.",
        parent=inst_node,
        critical=True
    )

    # Reference URLs verify claims (critical parallel group)
    refs_group = evaluator.add_parallel(
        id="Reference_URLs_Verify_Claims",
        desc="Provided reference URL(s) contain support for the claimed details (publication title, algorithm name, qubit count, speedup, and institution).",
        parent=inst_node,
        critical=True
    )

    # Child leaves for each claim
    url_title_node = evaluator.add_leaf(
        id="URL_Supports_Publication_Title",
        desc="At least one provided URL supports the stated publication title (e.g., the Nature article page).",
        parent=refs_group,
        critical=True
    )
    url_algo_node = evaluator.add_leaf(
        id="URL_Supports_Algorithm_Name",
        desc="At least one provided URL supports the stated algorithm implementation name.",
        parent=refs_group,
        critical=True
    )
    url_qubits_node = evaluator.add_leaf(
        id="URL_Supports_Qubit_Count",
        desc="At least one provided URL supports the stated qubit count (105).",
        parent=refs_group,
        critical=True
    )
    url_speed_node = evaluator.add_leaf(
        id="URL_Supports_Speedup",
        desc="At least one provided URL supports the stated speedup claim (~13,000×).",
        parent=refs_group,
        critical=True
    )
    url_inst_node = evaluator.add_leaf(
        id="URL_Supports_Institution",
        desc="At least one provided URL supports the stated primary institution (Google Quantum AI).",
        parent=refs_group,
        critical=True
    )

    # Build multi-URL verifications (will automatically use verify_by_urls if multiple provided)
    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = [
        (
            f"The publication title is '{EXPECTED['publication_title']}'.",
            urls,
            url_title_node,
            "Check the page for the exact title text. Prefer Nature or DOI pages if provided. Minor punctuation/case variants are acceptable only if the wording is identical."
        ),
        (
            f"The algorithm implementation is called '{EXPECTED['algorithm_impl_name']}'.",
            urls,
            url_algo_node,
            "Look for explicit mention of the implementation name; accept 'Quantum Echoes' only (not paraphrases)."
        ),
        (
            "The quantum processor uses exactly 105 qubits.",
            urls,
            url_qubits_node,
            "Accept forms like '105 qubits', '105‑qubit', or 'a 105-qubit processor'."
        ),
        (
            "The claimed speedup is approximately 13,000× compared to classical supercomputers.",
            urls,
            url_speed_node,
            "Accept notation variants like '13,000x', '~13k×', or 'roughly 13,000 times'."
        ),
        (
            "The primary institution responsible for the result is Google Quantum AI.",
            urls,
            url_inst_node,
            "Accept 'Google Quantum AI', 'Google Research, Quantum AI', or equivalent phrasings."
        ),
    ]

    # Execute URL verifications in parallel (will respect preconditions, e.g., missing URLs)
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the October 2025 verifiable quantum advantage Nature achievement.
    """
    # Initialize evaluator and root
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

    # Extract structured fields from the answer
    extraction: QuantumAchievementExtraction = await evaluator.extract(
        prompt=prompt_extract_quantum_achievement(),
        template_class=QuantumAchievementExtraction,
        extraction_name="quantum_achievement_extraction"
    )

    # Add ground-truth expectations to summary
    evaluator.add_ground_truth({
        "expected": EXPECTED,
        "notes": "All constrained fields must be met; URLs must support the claims."
    })

    # Add custom info: number of reference URLs extracted
    evaluator.add_custom_info(
        info={"reference_urls_count": len(extraction.reference_urls or [])},
        info_type="stats",
        info_name="extraction_statistics"
    )

    # Build the evaluation tree under a critical parent node
    qar_node = evaluator.add_parallel(
        id="Quantum_Achievement_Response",
        desc="Evaluate whether the response identifies the specified Nature (Oct 2025) verifiable-quantum-advantage achievement and provides all required fields consistent with constraints.",
        parent=root,
        critical=True
    )

    # Add subtrees (all critical children under a critical parent)
    await add_publication_details_checks(evaluator, qar_node)
    await add_algorithm_details_checks(evaluator, qar_node)
    await add_hardware_details_checks(evaluator, qar_node)
    await add_performance_checks(evaluator, qar_node)
    await add_institution_and_references_checks(evaluator, qar_node, extraction)

    # Return the structured evaluation summary
    return evaluator.get_summary()