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
TASK_ID = "quantum_platform_requirements"
TASK_DESCRIPTION = """Identify a cloud-accessible quantum computing platform that satisfies ALL of the following requirements:

Hardware & Technical Specifications:
1. Provides access to quantum processors with at least 20 physical qubits
2. Documents two-qubit gate fidelity of at least 99%
3. Supports quantum circuit execution with depth of at least 100 layers
4. For superconducting quantum processors, operates with proper cryogenic cooling systems (dilution refrigerators) maintaining millikelvin-range temperatures

Access & Programming:
5. Offers cloud-based access to actual quantum processing units (QPUs), not just simulators
6. Supports remote job submission via web interface or API
7. Supports at least one major quantum programming framework (Qiskit, Cirq, or Q#)
8. Provides a Python-based SDK or API
9. Includes circuit optimization and transpilation capabilities
10. Provides comprehensive API reference documentation with working code examples

Education & Certification:
11. Offers structured learning materials organized into courses or modules
12. Includes hands-on programming tutorials with executable code
13. Clearly documents mathematical prerequisites (e.g., linear algebra, complex numbers)
14. Offers a formal industry-recognized certification program with examination component

Accessibility & Performance:
15. Provides either free access tier or special academic pricing (no hardware purchase required)
16. Is available internationally in multiple countries/regions (not restricted to single country)
17. Reports at least one standard benchmark metric (Quantum Volume, CLOPS, or Algorithmic Qubits) with publicly documented values
18. Provides error mitigation techniques or tools in the software stack
19. Reports error rates or gate fidelities transparently

Infrastructure & Support:
20. Has documented partnerships or collaborations with academic institutions
21. Maintains active technical support channels and user community

Provide the name of the quantum computing platform and document each requirement with reference URLs from the platform's official documentation.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PlatformRequirementsExtraction(BaseModel):
    """Extraction of platform name and requirement-specific official URLs from the answer."""
    platform_name: Optional[str] = None

    # Hardware & Technical Specifications
    cloud_qpu_qubits_urls: List[str] = Field(default_factory=list)  # real QPUs + ≥20 qubits
    fidelity_urls: List[str] = Field(default_factory=list)          # two-qubit gate fidelity ≥99%
    circuit_depth_urls: List[str] = Field(default_factory=list)     # circuit depth ≥100
    cryo_urls: List[str] = Field(default_factory=list)              # dilution refrigerators, mK temps for superconducting
    hardware_modality_urls: List[str] = Field(default_factory=list) # pages describing hardware modality (superconducting / trapped-ion / photonic / neutral atoms)

    # Access & Programming
    remote_job_urls: List[str] = Field(default_factory=list)        # remote job submission (web UI or API)
    framework_urls: List[str] = Field(default_factory=list)         # supports Qiskit, Cirq, or Q#
    python_sdk_urls: List[str] = Field(default_factory=list)        # Python-based SDK/API
    optimization_urls: List[str] = Field(default_factory=list)      # optimization/transpilation capabilities
    api_reference_urls: List[str] = Field(default_factory=list)     # comprehensive API reference with code examples

    # Education & Certification
    learning_courses_urls: List[str] = Field(default_factory=list)  # structured courses/modules
    tutorials_urls: List[str] = Field(default_factory=list)         # hands-on tutorials with executable code
    math_prereq_urls: List[str] = Field(default_factory=list)       # math prerequisites documented
    certification_urls: List[str] = Field(default_factory=list)     # certification program with exam

    # Accessibility & Performance
    pricing_urls: List[str] = Field(default_factory=list)           # free tier or academic pricing, no hardware purchase
    international_urls: List[str] = Field(default_factory=list)     # international availability
    benchmark_urls: List[str] = Field(default_factory=list)         # metrics: Quantum Volume, CLOPS, Algorithmic Qubits
    error_mitigation_urls: List[str] = Field(default_factory=list)  # error mitigation tools/techniques
    error_transparency_urls: List[str] = Field(default_factory=list)# error rates/fidelities transparency

    # Infrastructure & Support
    academic_partnership_urls: List[str] = Field(default_factory=list) # academic partnerships/collaborations
    support_community_urls: List[str] = Field(default_factory=list)    # support channels & user community


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_platform_and_urls() -> str:
    return """
    Extract the name of the cloud-accessible quantum computing platform identified in the answer, and collect the official documentation URLs provided in the answer for each requirement below. Only include URLs that are explicitly present in the answer. Do not invent URLs.

    Official documentation means pages hosted by or directly maintained by the platform provider organization (e.g., company official site, official product docs, official blogs, or official GitHub of the provider). Avoid third-party news or blogs unless they are on an official domain.

    Fields to extract:
    - platform_name: The name of the quantum computing platform chosen in the answer.

    For each of the following fields, extract an array of official URLs the answer cites to support the requirement:

    Hardware & Technical Specifications:
    - cloud_qpu_qubits_urls: URLs showing cloud access to real QPUs (not only simulators) AND devices with ≥20 physical qubits
    - fidelity_urls: URLs documenting two-qubit gate fidelity ≥99%
    - circuit_depth_urls: URLs indicating circuits with depth ≥100 layers are supported
    - cryo_urls: URLs describing dilution refrigerators maintaining millikelvin-range temperatures (only relevant if superconducting QPUs are offered)
    - hardware_modality_urls: URLs describing the hardware modality offered (e.g., superconducting, trapped-ion, photonic, neutral atoms)

    Access & Programming:
    - remote_job_urls: URLs showing remote job submission via web UI or API
    - framework_urls: URLs showing support for at least one major framework (Qiskit, Cirq, or Q#)
    - python_sdk_urls: URLs showing a Python-based SDK or API
    - optimization_urls: URLs showing circuit optimization/transpilation capabilities
    - api_reference_urls: URLs showing comprehensive API reference with working code examples

    Education & Certification:
    - learning_courses_urls: URLs with structured learning courses/modules
    - tutorials_urls: URLs with hands-on programming tutorials (with executable code)
    - math_prereq_urls: URLs documenting mathematical prerequisites (e.g., linear algebra, complex numbers)
    - certification_urls: URLs describing a certification program with an exam component

    Accessibility & Performance:
    - pricing_urls: URLs showing either a free access tier or academic pricing (no hardware purchase required)
    - international_urls: URLs showing availability in multiple countries/regions
    - benchmark_urls: URLs reporting at least one standard benchmark metric (Quantum Volume, CLOPS, or Algorithmic Qubits) with public values
    - error_mitigation_urls: URLs documenting error mitigation techniques/tools in the software stack
    - error_transparency_urls: URLs transparently reporting error rates or gate fidelities

    Infrastructure & Support:
    - academic_partnership_urls: URLs documenting partnerships or collaborations with academic institutions
    - support_community_urls: URLs documenting active technical support channels and community

    Return null for platform_name if not stated. For each URL list, return an empty array if the answer does not provide any official URLs for that requirement.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


def merge_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    seen.add(u2)
                    merged.append(u2)
    return merged


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def add_requirement_with_sources_check(
    evaluator: Evaluator,
    parent_node,
    req_id: str,
    req_desc: str,
    urls: List[str],
    claim_text: str,
    additional_instruction: str,
    check_sources: bool = True
) -> None:
    """
    Create a critical sequential requirement node:
    1) A critical existence check that official URL(s) are provided (optional via check_sources)
    2) A critical verification leaf that checks the claim is supported by the provided official URLs
    """
    req_node = evaluator.add_sequential(
        id=req_id,
        desc=req_desc,
        parent=parent_node,
        critical=True
    )

    if check_sources:
        evaluator.add_custom_node(
            result=nonempty_urls(urls),
            id=f"{req_id}_sources_provided",
            desc="Official URL(s) are provided in the answer to support this requirement",
            parent=req_node,
            critical=True
        )

    verify_leaf = evaluator.add_leaf(
        id=f"{req_id}_supported_by_sources",
        desc=req_desc,
        parent=req_node,
        critical=True
    )

    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=urls,
        additional_instruction=additional_instruction
    )


async def build_verification_tree(evaluator: Evaluator, extraction: PlatformRequirementsExtraction) -> None:
    """
    Build the verification tree according to the rubric.
    Root is sequential: first ensure platform name is provided, then verify all requirements in parallel (all critical).
    """
    root = evaluator.root

    # 1) Platform name provided (critical)
    evaluator.add_custom_node(
        result=(extraction.platform_name is not None and extraction.platform_name.strip() != ""),
        id="platform_name_provided",
        desc="Answer provides the name of the quantum computing platform",
        parent=root,
        critical=True
    )

    # 2) Requirements satisfied and cited (critical parallel group)
    reqs_parent = evaluator.add_parallel(
        id="requirements_satisfied_and_cited",
        desc="Each required property is satisfied and supported by at least one official documentation URL",
        parent=root,
        critical=True
    )

    platform = extraction.platform_name or "the platform"

    # Hardware & Technical Specifications
    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "cloud_qpu_access_and_qubits",
        "Platform provides cloud-based access to real QPUs (not simulators-only) with ≥20 physical qubits, supported by official URL(s)",
        merge_urls(extraction.cloud_qpu_qubits_urls),
        claim_text=f"{platform} provides cloud-based access to actual QPUs (not only simulators) and offers at least one device with 20 or more physical qubits.",
        additional_instruction="Check that the documentation explicitly states access to real hardware via the cloud and that some listed device(s) have ≥20 physical qubits."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "two_qubit_gate_fidelity",
        "Official specs document two-qubit gate fidelity ≥99%, supported by official URL(s)",
        merge_urls(extraction.fidelity_urls),
        claim_text=f"{platform} documents a two-qubit gate fidelity of at least 99%.",
        additional_instruction="Accept statements like 'two-qubit fidelity ≥ 0.99', or equivalent error rates ≤1%. Ensure the value is clearly stated in official specs."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "circuit_depth",
        "Platform supports executing circuits with depth ≥100 layers, supported by official URL(s)",
        merge_urls(extraction.circuit_depth_urls),
        claim_text=f"{platform} supports executing quantum circuits with a depth of at least 100 layers.",
        additional_instruction="Look for documentation indicating circuit depth limits or examples demonstrating depth ≥100. If a hard cap below 100 exists, this should fail."
    )

    # Cryogenics for superconducting (conditional)
    # Use OR-style claim and combine hardware modality pages with cryo pages,
    # so verification can succeed if platform is not superconducting or,
    # if superconducting is offered, cryogenic dilution refrigerator details are documented.
    cryo_sources = merge_urls(extraction.cryo_urls, extraction.hardware_modality_urls)
    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "cryogenics_for_superconducting",
        "If the platform offers superconducting QPUs, official docs describe dilution-refrigerator cryogenic cooling maintaining millikelvin-range temperatures, supported by official URL(s)",
        cryo_sources,
        claim_text=(
            f"Either {platform} does not offer superconducting QPUs, "
            f"OR for superconducting devices it offers, its documentation describes dilution refrigerators "
            f"maintaining millikelvin-range operating temperatures."
        ),
        additional_instruction="Use hardware modality pages to verify non-superconducting technologies, or cryogenics documentation for superconducting devices (e.g., dilution refrigerator ~10 mK)."
    )

    # Access & Programming
    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "remote_job_submission",
        "Platform supports remote job submission via web interface or API, supported by official URL(s)",
        merge_urls(extraction.remote_job_urls),
        claim_text=f"{platform} supports remote job submission via a web interface or programmatic API.",
        additional_instruction="Look for user guides or APIs that describe job submission, queues, or run/submit commands."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "major_framework_support",
        "Platform supports at least one major framework: Qiskit, Cirq, or Q#, supported by official URL(s)",
        merge_urls(extraction.framework_urls),
        claim_text=f"{platform} supports at least one major quantum programming framework among Qiskit, Cirq, or Q#.",
        additional_instruction="Verify integration or compatibility documentation with any of Qiskit, Cirq, or Q#."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "python_sdk_api",
        "Platform provides a Python-based SDK or API, supported by official URL(s)",
        merge_urls(extraction.python_sdk_urls),
        claim_text=f"{platform} provides a Python-based SDK or API for programming and interacting with the service.",
        additional_instruction="Look for official Python packages, SDKs, or API docs showing Python usage."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "optimization_and_transpilation",
        "Platform includes circuit optimization and transpilation capabilities, supported by official URL(s)",
        merge_urls(extraction.optimization_urls),
        claim_text=f"{platform} provides circuit optimization and transpilation capabilities in its toolchain.",
        additional_instruction="Check docs for 'transpile', 'optimize', 'passes', 'compilation', or similar functionality."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "api_reference_with_examples",
        "Platform provides comprehensive API reference documentation with working code examples, supported by official URL(s)",
        merge_urls(extraction.api_reference_urls),
        claim_text=f"{platform} offers comprehensive API reference documentation that includes working code examples.",
        additional_instruction="Look for reference sections with code blocks or example snippets demonstrating API usage."
    )

    # Education & Certification
    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "structured_learning_materials",
        "Platform offers structured learning materials organized into courses or modules, supported by official URL(s)",
        merge_urls(extraction.learning_courses_urls),
        claim_text=f"{platform} provides structured learning materials organized into courses or modules.",
        additional_instruction="Verify structured curricula, course sequences, or learning paths provided by the platform."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "hands_on_tutorials",
        "Learning resources include hands-on programming tutorials with executable code, supported by official URL(s)",
        merge_urls(extraction.tutorials_urls),
        claim_text=f"{platform}'s learning resources include hands-on programming tutorials that feature executable code.",
        additional_instruction="Look for tutorials with step-by-step code blocks or notebooks that can be executed."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "math_prerequisites",
        "Platform clearly documents mathematical prerequisites (e.g., linear algebra, complex numbers), supported by official URL(s)",
        merge_urls(extraction.math_prereq_urls),
        claim_text=f"{platform} clearly documents mathematical prerequisites such as linear algebra and complex numbers.",
        additional_instruction="Check learning or documentation pages that list prerequisite mathematics."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "certification_with_exam",
        "Platform offers a formal industry-recognized certification program that includes an examination component, supported by official URL(s)",
        merge_urls(extraction.certification_urls),
        claim_text=f"{platform} offers a formal certification program that includes an examination component.",
        additional_instruction="Verify certification pages describing exams, credentials, or proctored assessments."
    )

    # Accessibility & Performance
    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "free_or_academic_pricing_no_hardware_purchase",
        "Platform provides either a free access tier or special academic pricing, with no hardware purchase required, supported by official URL(s)",
        merge_urls(extraction.pricing_urls),
        claim_text=f"{platform} provides either a free access tier or special academic pricing and does not require a hardware purchase.",
        additional_instruction="Check pricing or access pages that describe free tiers, grants, academic discounts, or subscription plans without buying hardware."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "international_availability",
        "Platform is available internationally in multiple countries/regions (not restricted to a single country), supported by official URL(s)",
        merge_urls(extraction.international_urls),
        claim_text=f"{platform} is available internationally in multiple countries or regions, not restricted to a single country.",
        additional_instruction="Look for service availability statements, region lists, or international rollout docs."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "benchmark_metric_with_public_values",
        "Platform reports at least one standard benchmark metric (Quantum Volume, CLOPS, or Algorithmic Qubits) with publicly documented values, supported by official URL(s)",
        merge_urls(extraction.benchmark_urls),
        claim_text=f"{platform} publicly reports at least one standard benchmark metric (such as Quantum Volume, CLOPS, or Algorithmic Qubits) with documented values.",
        additional_instruction="Verify the presence of metric names and actual values or charts in official documentation or announcements."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "error_mitigation_tools",
        "Platform provides error mitigation techniques or tools in its software stack, supported by official URL(s)",
        merge_urls(extraction.error_mitigation_urls),
        claim_text=f"{platform} provides error mitigation techniques or tools in its software stack.",
        additional_instruction="Look for features like measurement error mitigation, dynamical decoupling, zero-noise extrapolation, readout error mitigation, etc."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "error_rate_or_fidelity_transparency",
        "Platform reports error rates or gate fidelities transparently in public documentation, supported by official URL(s)",
        merge_urls(extraction.error_transparency_urls),
        claim_text=f"{platform} transparently reports error rates or gate fidelities in public documentation.",
        additional_instruction="Check device status pages, calibration reports, or documentation that lists gate or readout error rates/fidelities."
    )

    # Infrastructure & Support
    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "academic_partnerships",
        "Platform has documented partnerships/collaborations with academic institutions, supported by official URL(s)",
        merge_urls(extraction.academic_partnership_urls),
        claim_text=f"{platform} has documented partnerships or collaborations with academic institutions.",
        additional_instruction="Look for announcements, partner lists, or program pages highlighting university collaborations."
    )

    await add_requirement_with_sources_check(
        evaluator,
        reqs_parent,
        "support_and_user_community",
        "Platform maintains active technical support channels and a user community, supported by official URL(s)",
        merge_urls(extraction.support_community_urls),
        claim_text=f"{platform} maintains active technical support channels and a user community.",
        additional_instruction="Verify developer forums, Slack/Discord, support portals, helpdesks, GitHub issues, or similar official channels."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the cloud-accessible quantum platform requirements task.
    Builds a sequential root tree:
      - Critical platform name existence
      - Critical parallel set of requirement checks (each with source existence + URL-supported verification)
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # 1) Extract platform name and requirement URLs
    extraction = await evaluator.extract(
        prompt=prompt_extract_platform_and_urls(),
        template_class=PlatformRequirementsExtraction,
        extraction_name="platform_and_requirement_urls"
    )

    # 2) Build verification tree and run verifications
    await build_verification_tree(evaluator, extraction)

    # 3) Return structured summary
    return evaluator.get_summary()
