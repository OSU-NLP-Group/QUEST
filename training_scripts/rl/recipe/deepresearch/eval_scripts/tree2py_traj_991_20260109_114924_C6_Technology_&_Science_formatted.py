import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "quantum_hybrid_qaas_drug_2026"
TASK_DESCRIPTION = """
A pharmaceutical research company is planning to deploy hybrid quantum-classical computing infrastructure for drug discovery applications in 2026. They require quantum computing systems that can be accessed remotely via cloud platforms and integrated with their existing high-performance computing clusters. The systems must be capable of executing quantum algorithms for molecular simulation with sufficient circuit depth and reliability.

Identify four distinct quantum computing systems that are commercially available as of 2024-2025 and meet all of the following technical requirements:

1. The system must be accessible via cloud-based Quantum-as-a-Service (QaaS) platforms, without requiring on-premises installation of quantum hardware.

2. The system must provide at least 30 physical qubits to enable meaningful algorithmic complexity for drug discovery calculations.

3. The system must achieve a minimum two-qubit gate fidelity of 99% or higher, as documented in official vendor announcements or peer-reviewed publications.

4. If the system uses trapped-ion technology, it must provide all-to-all qubit connectivity to enable flexible quantum circuit compilation. If the system uses superconducting technology, it must achieve gate operation times of 100 nanoseconds or faster.

5. The system must have demonstrated quantum error correction capabilities or below-threshold performance in published benchmarks, indicating progress toward fault-tolerant quantum computing.

6. The system's coherence time (for superconducting qubits) or qubit stability must be sufficient to execute quantum circuits containing at least 100 two-qubit gates, which is necessary for the target molecular simulation algorithms.

7. The system must have achieved a documented performance milestone on at least one standard quantum computing benchmark, such as Quantum Volume (QV), Algorithmic Qubits (#AQ), or Random Circuit Sampling (RCS).

8. The vendor must provide documented integration capabilities with classical high-performance computing (HPC) infrastructure to enable hybrid quantum-classical workflows, which are essential for the variational quantum algorithms planned for drug discovery.

For each of the four systems you identify, provide:
- The system name and vendor
- Qubit technology type (trapped-ion or superconducting)
- Physical qubit count
- Two-qubit gate fidelity
- Key performance metrics (coherence time, gate speed, or benchmark scores)
- Cloud access platform/method
- Verification URLs from official vendor announcements or technical publications confirming each specification
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QuantumSystem(BaseModel):
    system_name: Optional[str] = None
    vendor: Optional[str] = None
    technology_type: Optional[str] = None  # Expected: trapped-ion or superconducting (synonyms acceptable)
    physical_qubits: Optional[str] = None  # Prefer string to capture ranges/text
    two_qubit_gate_fidelity: Optional[str] = None  # e.g., "99.5%", "≥99%"
    coherence_time: Optional[str] = None  # e.g., "100 us T1", "1 ms T2"
    gate_speed: Optional[str] = None  # e.g., "50 ns two-qubit", "100ns"
    benchmark_metrics: List[str] = Field(default_factory=list)  # e.g., ["QV 256", "#AQ=30"]
    cloud_access_platform: Optional[str] = None  # e.g., "IBM Quantum", "AWS Braket", "Azure Quantum"

    # Verification URLs by category
    availability_urls: List[str] = Field(default_factory=list)
    cloud_access_urls: List[str] = Field(default_factory=list)
    technology_urls: List[str] = Field(default_factory=list)
    qubit_count_urls: List[str] = Field(default_factory=list)
    fidelity_urls: List[str] = Field(default_factory=list)
    error_correction_urls: List[str] = Field(default_factory=list)
    coherence_urls: List[str] = Field(default_factory=list)
    benchmark_urls: List[str] = Field(default_factory=list)
    integration_urls: List[str] = Field(default_factory=list)


class SystemsExtraction(BaseModel):
    systems: List[QuantumSystem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_systems() -> str:
    return """
    Extract up to four quantum computing systems mentioned in the answer that meet the specified requirements for hybrid quantum-classical drug discovery applications (2024-2025). For each system, extract the following fields exactly as stated in the answer:

    - system_name: The product/system name
    - vendor: The company providing the system
    - technology_type: The qubit technology type. Use a simple canonical string such as "trapped-ion" or "superconducting" if present; if synonyms are used (e.g., "ion trap", "transmon"), still extract the provided term exactly.
    - physical_qubits: The physical qubit count (extract the numeric expression or textual form provided)
    - two_qubit_gate_fidelity: The reported two-qubit gate fidelity (e.g., "99.9%")
    - coherence_time: Any coherence time metric if given (e.g., "100 us", "1 ms")
    - gate_speed: Any gate speed/operation time metric if given (e.g., "100 ns two-qubit")
    - benchmark_metrics: A list of any benchmark achievements provided (e.g., "QV 256", "#AQ=30", "RCS")
    - cloud_access_platform: The cloud platform or method for QaaS access (e.g., "IBM Quantum", "AWS Braket", "Azure Quantum", "IonQ Cloud")

    Also extract verification URLs for each specification category if provided in the answer (only include actual URLs):
    - availability_urls: URLs confirming commercial availability (2024-2025)
    - cloud_access_urls: URLs confirming cloud-based access/QaaS
    - technology_urls: URLs confirming the technology type/specifications
    - qubit_count_urls: URLs confirming the physical qubit count
    - fidelity_urls: URLs confirming the two-qubit gate fidelity
    - error_correction_urls: URLs confirming error correction capabilities or below-threshold performance
    - coherence_urls: URLs supporting coherence/stability sufficient for circuits with ≥100 two-qubit gates
    - benchmark_urls: URLs confirming benchmark achievements (QV, #AQ, RCS)
    - integration_urls: URLs confirming integration with classical HPC infrastructure

    Return a JSON object with a single key "systems" that is an array of system objects with the above fields. If a field is missing for a system, set it to null (or empty list for arrays). Extract only URLs explicitly present in the answer text. Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Helper parsing functions                                                    #
# --------------------------------------------------------------------------- #
def parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    # Choose the maximum encountered integer to be conservative
    return max(int(n) for n in nums)


def parse_percent_to_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%?", text)
    if not matches:
        return None
    # Use the maximum percentage found
    return max(float(m) for m in matches)


def classify_technology(tech_text: Optional[str]) -> Optional[str]:
    if not tech_text:
        return None
    t = tech_text.lower()
    if any(k in t for k in ["trapped-ion", "ion trap", "ion-trap", "iontrap", "trapped ion"]):
        return "trapped-ion"
    if any(k in t for k in ["superconduct", "transmon", "superconducting"]):
        return "superconducting"
    return None


def union_urls(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for u in lst:
            if isinstance(u, str) and u.strip():
                if u not in seen:
                    seen.add(u)
                    result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification logic for one system                                           #
# --------------------------------------------------------------------------- #
async def verify_system(
    evaluator: Evaluator,
    parent_node,
    sys: QuantumSystem,
    index: int,
) -> None:
    # System container node
    sys_node = evaluator.add_parallel(
        id=f"system_{index + 1}",
        desc=f"{['First','Second','Third','Fourth'][index]} quantum computing system meeting all specified requirements",
        parent=parent_node,
        critical=False,
    )

    # ---------------------- Commercial availability ---------------------- #
    avail_node = evaluator.add_parallel(
        id=f"system_{index + 1}_commercial_availability",
        desc="System is commercially available via cloud access in 2024-2025",
        parent=sys_node,
        critical=True,
    )

    # Leaf: availability_status
    availability_status_leaf = evaluator.add_leaf(
        id=f"system_{index + 1}_availability_status",
        desc="System is officially announced as commercially available by the vendor",
        parent=avail_node,
        critical=True,
    )
    availability_sources = union_urls(sys.availability_urls)
    await evaluator.verify(
        claim=f"The system '{sys.system_name or ''}' by '{sys.vendor or ''}' is commercially available (announced/accessible) in 2024–2025.",
        node=availability_status_leaf,
        sources=availability_sources,
        additional_instruction="Confirm via official vendor sources (product pages, announcements, press releases) that the system is commercially available in the 2024–2025 timeframe.",
    )

    # Leaf: cloud_access
    cloud_access_leaf = evaluator.add_leaf(
        id=f"system_{index + 1}_cloud_access",
        desc="System provides cloud-based access (QaaS) without requiring on-premises installation",
        parent=avail_node,
        critical=True,
    )
    cloud_sources = union_urls(sys.cloud_access_urls, sys.availability_urls)
    cloud_platform_text = sys.cloud_access_platform or ""
    await evaluator.verify(
        claim=f"The system provides cloud-based QaaS access (e.g., via '{cloud_platform_text}') and does not require on-premises quantum hardware.",
        node=cloud_access_leaf,
        sources=cloud_sources,
        additional_instruction="Confirm the QaaS/cloud access method (e.g., IBM Quantum, AWS Braket, Azure Quantum, IonQ Cloud) and that no on-prem hardware is required.",
    )

    # Leaf: availability_reference (presence of URL)
    evaluator.add_custom_node(
        result=len(availability_sources) > 0,
        id=f"system_{index + 1}_availability_reference",
        desc="URL reference confirming commercial availability",
        parent=avail_node,
        critical=True,
    )

    # ---------------------- Qubit technology ----------------------------- #
    tech_node = evaluator.add_sequential(
        id=f"system_{index + 1}_qubit_technology",
        desc="Qubit technology type and architecture specifications",
        parent=sys_node,
        critical=True,
    )

    # Step 1: technology_identification (critical)
    canonical_tech = classify_technology(sys.technology_type)
    evaluator.add_custom_node(
        result=canonical_tech in ("trapped-ion", "superconducting"),
        id=f"system_{index + 1}_technology_identification",
        desc="Technology type (trapped-ion or superconducting) is clearly identified",
        parent=tech_node,
        critical=True,
    )

    # Step 2: technology_specific_requirements (parallel, critical)
    tech_spec_node = evaluator.add_parallel(
        id=f"system_{index + 1}_technology_specific_requirements",
        desc="Technology-specific performance requirements are met",
        parent=tech_node,
        critical=True,
    )

    if canonical_tech == "trapped-ion":
        ion_connect_leaf = evaluator.add_leaf(
            id=f"system_{index + 1}_trapped_ion_connectivity",
            desc="If trapped-ion: provides all-to-all qubit connectivity",
            parent=tech_spec_node,
            critical=True,
        )
        await evaluator.verify(
            claim="This trapped-ion system provides all-to-all qubit connectivity.",
            node=ion_connect_leaf,
            sources=union_urls(sys.technology_urls),
            additional_instruction="Verify all-to-all connectivity from vendor or technical documentation.",
        )
    elif canonical_tech == "superconducting":
        sc_gate_leaf = evaluator.add_leaf(
            id=f"system_{index + 1}_superconducting_gate_speed",
            desc="If superconducting: achieves gate operation times <=100ns",
            parent=tech_spec_node,
            critical=True,
        )
        await evaluator.verify(
            claim="This superconducting system achieves two-qubit gate operation times of 100 nanoseconds or faster.",
            node=sc_gate_leaf,
            sources=union_urls(sys.technology_urls),
            additional_instruction="Verify gate operation times <=100ns (prefer two-qubit gate speeds).",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"system_{index + 1}_technology_specific_path_selected",
            desc="Technology-specific requirement path can be selected (trapped-ion or superconducting)",
            parent=tech_spec_node,
            critical=True,
        )

    # Step 3: technology_reference (critical)
    tech_ref_leaf = evaluator.add_leaf(
        id=f"system_{index + 1}_technology_reference",
        desc="URL reference confirming technology specifications",
        parent=tech_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The specified technology type ('{sys.technology_type or ''}') for '{sys.system_name or ''}' is confirmed by vendor documentation.",
        node=tech_ref_leaf,
        sources=union_urls(sys.technology_urls),
        additional_instruction="Confirm that vendor/technical documentation clearly supports the stated technology type and architecture.",
    )

    # ---------------------- Qubit count ---------------------------------- #
    qubits_node = evaluator.add_parallel(
        id=f"system_{index + 1}_qubit_count",
        desc="Physical qubit count meets minimum requirements",
        parent=sys_node,
        critical=True,
    )

    # Leaf: minimum_qubits (critical, custom)
    qubits_num = parse_int_from_text(sys.physical_qubits)
    evaluator.add_custom_node(
        result=(qubits_num is not None and qubits_num >= 30),
        id=f"system_{index + 1}_minimum_qubits",
        desc="System provides ≥30 physical qubits",
        parent=qubits_node,
        critical=True,
    )

    # Leaf: qubit_count_verification
    qubits_verify_leaf = evaluator.add_leaf(
        id=f"system_{index + 1}_qubit_count_verification",
        desc="Qubit count is verified from official vendor specifications",
        parent=qubits_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The system provides {sys.physical_qubits or ''} physical qubits (meeting or exceeding 30).",
        node=qubits_verify_leaf,
        sources=union_urls(sys.qubit_count_urls),
        additional_instruction="Verify the physical qubit count from vendor specs or official documentation.",
    )

    # Leaf: qubit_count_reference (presence)
    evaluator.add_custom_node(
        result=len(sys.qubit_count_urls) > 0,
        id=f"system_{index + 1}_qubit_count_reference",
        desc="URL reference confirming qubit count",
        parent=qubits_node,
        critical=True,
    )

    # ---------------------- Gate fidelity -------------------------------- #
    fidelity_node = evaluator.add_parallel(
        id=f"system_{index + 1}_gate_fidelity",
        desc="Two-qubit gate fidelity performance",
        parent=sys_node,
        critical=True,
    )

    # Leaf: fidelity_threshold (custom)
    fidelity_val = parse_percent_to_float(sys.two_qubit_gate_fidelity)
    evaluator.add_custom_node(
        result=(fidelity_val is not None and fidelity_val >= 99.0),
        id=f"system_{index + 1}_fidelity_threshold",
        desc="Two-qubit gate fidelity ≥99%",
        parent=fidelity_node,
        critical=True,
    )

    # Leaf: fidelity_measurement
    fidelity_measure_leaf = evaluator.add_leaf(
        id=f"system_{index + 1}_fidelity_measurement",
        desc="Fidelity measurement is from official vendor announcement or peer-reviewed publication",
        parent=fidelity_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The two-qubit gate fidelity of at least 99% is documented in an official vendor announcement or peer-reviewed publication.",
        node=fidelity_measure_leaf,
        sources=union_urls(sys.fidelity_urls),
        additional_instruction="Evaluate whether the provided URL(s) are official vendor sources or peer-reviewed publications and they document ≥99% two-qubit fidelity.",
    )

    # Leaf: fidelity_reference (presence)
    evaluator.add_custom_node(
        result=len(sys.fidelity_urls) > 0,
        id=f"system_{index + 1}_fidelity_reference",
        desc="URL reference confirming gate fidelity",
        parent=fidelity_node,
        critical=True,
    )

    # ---------------------- Error correction ----------------------------- #
    ec_node = evaluator.add_parallel(
        id=f"system_{index + 1}_error_correction",
        desc="Error correction or below-threshold performance capabilities",
        parent=sys_node,
        critical=True,
    )

    # Leaf: error_correction_demonstration
    ec_demo_leaf = evaluator.add_leaf(
        id=f"system_{index + 1}_error_correction_demonstration",
        desc="System has demonstrated error correction capabilities or below-threshold performance",
        parent=ec_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The system has demonstrated quantum error correction capabilities or below-threshold performance.",
        node=ec_demo_leaf,
        sources=union_urls(sys.error_correction_urls),
        additional_instruction="Look for vendor or scientific publications about logical qubits, error suppression below threshold, QEC code demonstrations (e.g., surface/stabilizer/repetition codes).",
    )

    # Leaf: error_correction_reference (presence)
    evaluator.add_custom_node(
        result=len(sys.error_correction_urls) > 0,
        id=f"system_{index + 1}_error_correction_reference",
        desc="URL reference confirming error correction capabilities",
        parent=ec_node,
        critical=True,
    )

    # ---------------------- Circuit depth / coherence -------------------- #
    circ_node = evaluator.add_parallel(
        id=f"system_{index + 1}_circuit_depth",
        desc="System can execute sufficiently deep circuits for drug discovery applications",
        parent=sys_node,
        critical=True,
    )

    # Leaf: coherence_sufficiency
    coherence_leaf = evaluator.add_leaf(
        id=f"system_{index + 1}_coherence_sufficiency",
        desc="Coherence time or qubit stability allows execution of circuits with ≥100 two-qubit gates",
        parent=circ_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The system's coherence time or qubit stability is sufficient to execute circuits with at least 100 two-qubit gates.",
        node=coherence_leaf,
        sources=union_urls(sys.coherence_urls, sys.technology_urls),
        additional_instruction="Accept either explicit demonstrations of ≥100 two-qubit gates or coherence/gate-time metrics that reasonably imply such circuit depth.",
    )

    # Leaf: coherence_reference (presence)
    evaluator.add_custom_node(
        result=len(sys.coherence_urls) > 0,
        id=f"system_{index + 1}_coherence_reference",
        desc="URL reference confirming coherence time or stability metrics",
        parent=circ_node,
        critical=True,
    )

    # ---------------------- Benchmark performance ------------------------ #
    bench_node = evaluator.add_parallel(
        id=f"system_{index + 1}_benchmark_performance",
        desc="Documented performance on standard quantum computing benchmarks",
        parent=sys_node,
        critical=True,
    )

    # Leaf: benchmark_achievement
    bench_ach_leaf = evaluator.add_leaf(
        id=f"system_{index + 1}_benchmark_achievement",
        desc="System has achieved a documented milestone on at least one standard benchmark (QV, AQ, or RCS)",
        parent=bench_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The system has achieved a documented milestone on at least one standard benchmark: Quantum Volume (QV), Algorithmic Qubits (#AQ), or Random Circuit Sampling (RCS).",
        node=bench_ach_leaf,
        sources=union_urls(sys.benchmark_urls),
        additional_instruction="Verify benchmark achievement from vendor releases or publications; acceptable examples include QV scores, #AQ values, or RCS demonstrations.",
    )

    # Leaf: benchmark_reference (presence)
    evaluator.add_custom_node(
        result=len(sys.benchmark_urls) > 0,
        id=f"system_{index + 1}_benchmark_reference",
        desc="URL reference confirming benchmark performance",
        parent=bench_node,
        critical=True,
    )

    # ---------------------- Hybrid integration (HPC) --------------------- #
    hpc_node = evaluator.add_parallel(
        id=f"system_{index + 1}_hybrid_integration",
        desc="Integration capabilities with classical HPC infrastructure",
        parent=sys_node,
        critical=True,
    )

    # Leaf: integration_capability
    integration_leaf = evaluator.add_leaf(
        id=f"system_{index + 1}_integration_capability",
        desc="Vendor provides documented integration capabilities with classical HPC systems for hybrid workflows",
        parent=hpc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The vendor provides documented integration capabilities with classical HPC infrastructure enabling hybrid quantum-classical workflows.",
        node=integration_leaf,
        sources=union_urls(sys.integration_urls),
        additional_instruction="Confirm references to HPC cluster integration, workflow orchestration, hybrid VQA pipelines, or enterprise/HPC connectors.",
    )

    # Leaf: integration_reference (presence)
    evaluator.add_custom_node(
        result=len(sys.integration_urls) > 0,
        id=f"system_{index + 1}_integration_reference",
        desc="URL reference confirming hybrid integration capabilities",
        parent=hpc_node,
        critical=True,
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates systems independently
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

    # Extract systems
    extracted = await evaluator.extract(
        prompt=prompt_extract_systems(),
        template_class=SystemsExtraction,
        extraction_name="systems_extraction",
    )

    # Limit to first 4 systems; pad placeholders if fewer
    systems: List[QuantumSystem] = list(extracted.systems[:4])
    while len(systems) < 4:
        systems.append(QuantumSystem())

    # Build verification subtree for each system
    for idx in range(4):
        await verify_system(evaluator, root, systems[idx], idx)

    return evaluator.get_summary()
