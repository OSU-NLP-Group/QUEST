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
TASK_ID = "quantum_processor_specs"
TASK_DESCRIPTION = (
    "A quantum chemistry research laboratory is seeking to identify a commercially available quantum processor for "
    "molecular simulation studies. The processor must meet ALL of the following technical requirements: "
    "(1) two-qubit gate fidelity exceeding 99.9%, (2) at least 90 physical qubits, (3) all-to-all qubit connectivity "
    "architecture, (4) T2 coherence time of at least 500 milliseconds, (5) trapped-ion qubit technology, "
    "(6) demonstrated logical qubit capability with error correction, and (7) commercial availability with cloud access. "
    "Identify one quantum processor that satisfies all these requirements. Provide the processor name, manufacturer, "
    "and reference URLs documenting each specification."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class QuantumProcessorExtraction(BaseModel):
    """Structured information extracted from the agent's answer for the quantum processor task."""
    processor_name: Optional[str] = None
    manufacturer: Optional[str] = None

    # Claimed specs (flexible strings to handle natural language, ranges, etc.)
    two_qubit_fidelity: Optional[str] = None
    physical_qubits: Optional[str] = None
    connectivity: Optional[str] = None
    t2_coherence: Optional[str] = None
    qubit_tech: Optional[str] = None
    logical_qubit_capability: Optional[str] = None
    availability: Optional[str] = None
    cloud_access: Optional[str] = None

    # Source URLs per requirement
    fidelity_urls: List[str] = Field(default_factory=list)
    qubit_count_urls: List[str] = Field(default_factory=list)
    connectivity_urls: List[str] = Field(default_factory=list)
    t2_urls: List[str] = Field(default_factory=list)
    technology_urls: List[str] = Field(default_factory=list)
    logical_qubit_urls: List[str] = Field(default_factory=list)
    commercial_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_processor_info() -> str:
    return """
    You must extract the details for exactly one quantum processor that the answer identifies as satisfying all seven requirements.
    If multiple processors are mentioned, select the main one the answer ultimately recommends or emphasizes; otherwise select the first one.

    Extract the following fields:
    1. processor_name: The specific processor name/model identified (string; null if missing).
    2. manufacturer: The vendor/manufacturer of the processor (string; null if missing).

    For each of the seven requirements, extract BOTH the textual claim (if present) and all supporting reference URLs explicitly provided in the answer:
    3. two_qubit_fidelity: The stated two-qubit gate fidelity figure/description (string; null if missing).
    4. fidelity_urls: Array of URLs cited that support the two-qubit gate fidelity claim. If none provided, return an empty array.

    5. physical_qubits: The stated physical qubit count figure/description (string; null if missing).
    6. qubit_count_urls: Array of URLs cited that support the qubit count claim. If none provided, return an empty array.

    7. connectivity: The stated connectivity architecture description (e.g., "all-to-all connectivity") (string; null if missing).
    8. connectivity_urls: Array of URLs cited that support the connectivity claim. If none provided, return an empty array.

    9. t2_coherence: The stated T2 coherence time figure/description (string; null if missing).
    10. t2_urls: Array of URLs cited that support the T2 coherence claim. If none provided, return an empty array.

    11. qubit_tech: The stated qubit technology (e.g., "trapped-ion") (string; null if missing).
    12. technology_urls: Array of URLs cited that support the technology claim. If none provided, return an empty array.

    13. logical_qubit_capability: The stated logical qubit / error-correction capability description (string; null if missing).
    14. logical_qubit_urls: Array of URLs cited that support the logical qubit / error-correction demonstration. If none provided, return an empty array.

    15. availability: The stated commercial availability description (string; null if missing).
    16. cloud_access: The stated cloud access description (string; null if missing).
    17. commercial_urls: Array of URLs cited that support commercial availability and/or cloud access. If none provided, return an empty array.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer (plain URLs or markdown links). Do not invent or infer URLs.
    - Return full URLs. If a URL appears without protocol, prepend "http://".
    - If a requirement has multiple supporting URLs, include all of them in the corresponding array.
    - If a requirement has no supporting URLs, return an empty array for that field.

    IMPORTANT:
    - Do not include any URLs that are not clearly associated with the specific processor identified.
    - Do not include generic citations unless the answer explicitly uses them as sources for the processor’s specs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def anchor_prefix(info: QuantumProcessorExtraction) -> str:
    """
    Build an anchoring prefix for claims using the processor name and manufacturer when available.
    """
    name = (info.processor_name or "").strip()
    manu = (info.manufacturer or "").strip()

    if name and manu:
        return f"The processor {name} by {manu}"
    if name:
        return f"The processor {name}"
    if manu:
        return f"The processor by {manu}"
    return "The identified processor"


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root: Any, info: QuantumProcessorExtraction) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    Root created by Evaluator.initialize is non-critical; we add a critical top-level node to gate all requirements.
    """
    # Top-level critical container to gate all mandatory requirements
    task_main = evaluator.add_parallel(
        id="task_main",
        desc="Identify one commercially available quantum processor that meets all specified technical requirements, and provide processor name, manufacturer, and supporting reference URLs for each requirement.",
        parent=root,
        critical=True,
    )

    # 1) Processor identification (critical)
    identification_node = evaluator.add_parallel(
        id="processor_identification",
        desc="Provide processor identification details requested by the question.",
        parent=task_main,
        critical=True,
    )

    # 1.a Processor name exists (critical)
    evaluator.add_custom_node(
        result=bool(info.processor_name and info.processor_name.strip()),
        id="processor_name",
        desc="Processor name/model is provided.",
        parent=identification_node,
        critical=True,
    )

    # 1.b Manufacturer exists (critical)
    evaluator.add_custom_node(
        result=bool(info.manufacturer and info.manufacturer.strip()),
        id="manufacturer",
        desc="Manufacturer/vendor is provided.",
        parent=identification_node,
        critical=True,
    )

    # Anchor prefix for claims
    prefix = anchor_prefix(info)

    # Utility to add requirement nodes with a single condition leaf and a supporting URL existence check
    async def add_requirement_with_claim(
        req_id: str,
        req_desc: str,
        condition_desc: str,
        claim: str,
        urls: List[str],
        additional_instruction: str,
    ):
        node = evaluator.add_parallel(
            id=req_id,
            desc=req_desc,
            parent=task_main,
            critical=True,
        )
        # Supporting URL existence check first (critical)
        evaluator.add_custom_node(
            result=bool(urls and len(urls) > 0),
            id=f"{req_id}_supporting_url",
            desc=f"A reference URL is provided that documents the specification for {req_id.replace('requirement_', '').replace('_', ' ')}.",
            parent=node,
            critical=True,
        )
        # Condition verification leaf (critical)
        cond_node = evaluator.add_leaf(
            id=f"{req_id}_condition_met",
            desc=condition_desc,
            parent=node,
            critical=True,
        )
        await evaluator.verify(
            claim=claim,
            node=cond_node,
            sources=urls,  # Verify against the provided URLs (multi-URL verification supported)
            additional_instruction=additional_instruction,
        )

    # 2) Requirement 1: Two-qubit gate fidelity > 99.9%
    fidelity_claim = f"{prefix} has a two-qubit gate fidelity exceeding 99.9%."
    await add_requirement_with_claim(
        req_id="requirement_1_two_qubit_fidelity",
        req_desc="Two-qubit gate fidelity exceeds 99.9%, with supporting reference URL.",
        condition_desc="Evidence supports that two-qubit gate fidelity is > 99.9%.",
        claim=fidelity_claim,
        urls=info.fidelity_urls,
        additional_instruction=(
            "Confirm the page states a two-qubit gate fidelity greater than 99.9% (i.e., > 0.999). "
            "Accept reasonable equivalents, including '≥ 99.9%' or any value > 99.9% (e.g., 99.95%, 99.99%). "
            "If fidelity figures are reported for the specified processor family or exact model, that is acceptable."
        ),
    )

    # 3) Requirement 2: At least 90 physical qubits
    qubits_claim = f"{prefix} has at least 90 physical qubits."
    await add_requirement_with_claim(
        req_id="requirement_2_qubit_count",
        req_desc="At least 90 physical qubits, with supporting reference URL.",
        condition_desc="Evidence supports that the processor has >= 90 physical qubits.",
        claim=qubits_claim,
        urls=info.qubit_count_urls,
        additional_instruction=(
            "Verify the number of physical qubits is at least 90. Accept '≥ 90', 'at least 90', or any count >= 90. "
            "If the page lists a range or minimum that includes 90 or more, that is acceptable."
        ),
    )

    # 4) Requirement 3: All-to-all connectivity
    connectivity_claim = f"{prefix} provides all-to-all qubit connectivity."
    await add_requirement_with_claim(
        req_id="requirement_3_connectivity",
        req_desc="All-to-all qubit connectivity, with supporting reference URL.",
        condition_desc="Evidence supports that the architecture provides all-to-all connectivity.",
        claim=connectivity_claim,
        urls=info.connectivity_urls,
        additional_instruction=(
            "Verify the page explicitly indicates 'all-to-all connectivity' or equivalent wording (e.g., 'full connectivity', "
            "'each qubit can interact with any other qubit'). Minor wording variations are acceptable."
        ),
    )

    # 5) Requirement 4: T2 coherence time ≥ 500 ms
    t2_claim = f"{prefix} has a T2 coherence time of at least 500 milliseconds."
    await add_requirement_with_claim(
        req_id="requirement_4_t2_coherence",
        req_desc="T2 coherence time is at least 500 ms, with supporting reference URL.",
        condition_desc="Evidence supports that T2 coherence time is >= 500 milliseconds for the identified processor (or its documented operating specification).",
        claim=t2_claim,
        urls=info.t2_urls,
        additional_instruction=(
            "500 milliseconds equals 0.5 seconds. Accept any T2 coherence figure ≥ 0.5 s (e.g., 1 s, 10 s, minutes). "
            "If T2 is documented for the processor family or same technology variant, it's acceptable."
        ),
    )

    # 6) Requirement 5: Trapped-ion technology
    tech_claim = f"{prefix} uses trapped-ion qubit technology."
    await add_requirement_with_claim(
        req_id="requirement_5_trapped_ion",
        req_desc="Processor uses trapped-ion qubit technology, with supporting reference URL.",
        condition_desc="Evidence supports that the processor technology is trapped-ion.",
        claim=tech_claim,
        urls=info.technology_urls,
        additional_instruction=(
            "Verify the page explicitly states the processor uses trapped-ion qubits or ion-trap technology."
        ),
    )

    # 7) Requirement 6: Logical qubit capability with error correction demonstrated
    logical_claim = f"{prefix} has demonstrated logical qubit capability with error correction."
    await add_requirement_with_claim(
        req_id="requirement_6_logical_qubits_error_correction",
        req_desc="Demonstrated logical qubit capability with error correction, with supporting reference URL.",
        condition_desc="Evidence supports that logical qubit capability with error correction has been demonstrated.",
        claim=logical_claim,
        urls=info.logical_qubit_urls,
        additional_instruction=(
            "Verify that the page shows a demonstration of logical qubit(s) with quantum error correction on this processor or a closely related hardware variant "
            "by the same manufacturer. Accept demonstrations on the same platform family."
        ),
    )

    # 8) Requirement 7: Commercial availability with cloud access (two condition leaves)
    req7 = evaluator.add_parallel(
        id="requirement_7_commercial_cloud",
        desc="Commercial availability with cloud access, with supporting reference URL.",
        parent=task_main,
        critical=True,
    )

    # Supporting URL presence (critical)
    evaluator.add_custom_node(
        result=bool(info.commercial_urls and len(info.commercial_urls) > 0),
        id="requirement_7_commercial_cloud_supporting_url",
        desc="A reference URL is provided that documents commercial availability and/or cloud access.",
        parent=req7,
        critical=True,
    )

    # Commercial availability verification (critical)
    commercial_node = evaluator.add_leaf(
        id="requirement_7_commercial_availability_met",
        desc="Evidence supports that the processor is commercially available.",
        parent=req7,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{prefix} is commercially available.",
        node=commercial_node,
        sources=info.commercial_urls,
        additional_instruction=(
            "Confirm the page indicates commercial availability (e.g., purchasable, bookable time, enterprise access, or official listing for commercial use). "
            "Marketing pages that clearly state commercial availability or booking options are acceptable."
        ),
    )

    # Cloud access verification (critical)
    cloud_node = evaluator.add_leaf(
        id="requirement_7_cloud_access_met",
        desc="Evidence supports that the processor is accessible via cloud.",
        parent=req7,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{prefix} is accessible via cloud.",
        node=cloud_node,
        sources=info.commercial_urls,
        additional_instruction=(
            "Confirm the page indicates cloud access (e.g., accessible via vendor cloud, AWS Braket, Azure, or similar cloud services). "
            "Explicit mention of cloud platform integration or API-based remote access is acceptable."
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
    Evaluate an answer for the quantum processor specifications task.
    """
    # Initialize evaluator with a parallel root (non-critical root; we'll add a critical main node)
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
        default_model=model,
    )

    # Extract processor and URLs information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_processor_info(),
        template_class=QuantumProcessorExtraction,
        extraction_name="processor_extraction",
    )

    # Record the requirement checklist as ground truth context (no specific device ground truth)
    evaluator.add_ground_truth({
        "mandatory_requirements": [
            "Two-qubit gate fidelity > 99.9%",
            ">= 90 physical qubits",
            "All-to-all qubit connectivity",
            "T2 coherence time >= 500 ms",
            "Trapped-ion qubit technology",
            "Demonstrated logical qubit capability with error correction",
            "Commercial availability with cloud access",
        ],
        "note": "All requirements are mandatory; failure of any requirement fails the overall task."
    })

    # Add custom info for quick view of extracted processor details
    evaluator.add_custom_info(
        info={
            "processor_name": extraction.processor_name,
            "manufacturer": extraction.manufacturer,
            "sources": {
                "fidelity_urls": extraction.fidelity_urls,
                "qubit_count_urls": extraction.qubit_count_urls,
                "connectivity_urls": extraction.connectivity_urls,
                "t2_urls": extraction.t2_urls,
                "technology_urls": extraction.technology_urls,
                "logical_qubit_urls": extraction.logical_qubit_urls,
                "commercial_urls": extraction.commercial_urls,
            }
        },
        info_type="extraction_summary",
        info_name="extracted_processor_summary"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()