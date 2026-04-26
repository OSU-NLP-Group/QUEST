import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dual_quantum_infrastructure_2025"
TASK_DESCRIPTION = (
    "A quantum computing research facility is establishing a dual-system quantum computing infrastructure to support both "
    "fundamental quantum error correction research and practical quantum algorithm development. They need to identify two "
    "commercial quantum computer systems available in 2025 that meet the following requirements:\n\n"
    "System A — Ultra-High-Fidelity System for Error Correction Research:\n"
    "- Must use trapped-ion qubit technology\n"
    "- Must achieve two-qubit gate fidelity exceeding 99.9%\n"
    "- Must have at least 50 physical qubits\n"
    "- Must provide all-to-all qubit connectivity\n"
    "- Must have demonstrated quantum volume of at least 2^20 (1,048,576)\n"
    "- Must support mid-circuit measurement capability\n"
    "- Must be a commercially available system with official documentation\n\n"
    "System B — Mid-Scale Development System for Algorithm Development:\n"
    "- Must have between 50 and 150 physical qubits\n"
    "- Must achieve median two-qubit gate fidelity of at least 99%\n"
    "- Must be available for on-premise deployment\n"
    "- Must have a floor footprint not exceeding 10 square meters (approximately 6.8 m² or less)\n"
    "- Must have typical power consumption not exceeding 30 kilowatts\n"
    "- Must support surface code error correction architecture natively\n"
    "- Must be a commercially available system with official documentation\n\n"
    "For each system, identify the specific quantum computer model and manufacturer, and provide the official product "
    "documentation URL that confirms all the required specifications."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QuantumSystemA(BaseModel):
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    documentation_urls: List[str] = Field(default_factory=list)

    qubit_technology: Optional[str] = None                 # e.g., "trapped-ion"
    two_qubit_fidelity: Optional[str] = None               # e.g., "99.95%" or ">=99.9%"
    qubit_count: Optional[str] = None                      # e.g., "50", "64"
    connectivity: Optional[str] = None                     # e.g., "all-to-all", "full connectivity"
    quantum_volume: Optional[str] = None                   # e.g., "2^20", "1,048,576"
    mid_circuit_measurement: Optional[str] = None          # e.g., "supported", description text
    commercially_available_2025: Optional[str] = None      # Any explicit mention of availability in 2025


class QuantumSystemB(BaseModel):
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    documentation_urls: List[str] = Field(default_factory=list)

    qubit_count: Optional[str] = None                      # e.g., "100"
    two_qubit_fidelity_median: Optional[str] = None        # e.g., "99%", ">=99%"
    on_premise_deployment: Optional[str] = None            # e.g., "on-premise", "on-site", "on premises"
    floor_footprint: Optional[str] = None                  # e.g., "6.8 m^2", "73 sq ft"
    power_consumption: Optional[str] = None                # e.g., "25 kW"
    error_correction_support: Optional[str] = None         # e.g., "surface code", "native surface code"
    commercially_available_2025: Optional[str] = None


class DualSystemExtraction(BaseModel):
    system_a: Optional[QuantumSystemA] = None
    system_b: Optional[QuantumSystemB] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dual_systems() -> str:
    return """
    Extract the details of two commercial quantum computer systems (System A and System B) as presented in the answer.

    For System A (ultra-high-fidelity system for error correction research), extract:
    - model: Specific product/model name of the quantum computer
    - manufacturer: Company/manufacturer name
    - documentation_urls: A list of official manufacturer/product documentation URLs explicitly mentioned in the answer. Include product pages, datasheets, brochures, technical documentation pages. Do NOT invent URLs.
    - qubit_technology: The qubit technology (e.g., trapped-ion)
    - two_qubit_fidelity: The stated two-qubit gate fidelity (text exactly as presented)
    - qubit_count: The number of physical qubits (text exactly as presented)
    - connectivity: Description of qubit connectivity (e.g., all-to-all/full connectivity)
    - quantum_volume: Stated quantum volume (text exactly as presented), e.g., "2^20" or "1,048,576"
    - mid_circuit_measurement: Statement indicating whether mid-circuit measurement is supported
    - commercially_available_2025: Any statement indicating commercial availability in 2025 (text exactly as presented)

    For System B (mid-scale development system for algorithm development), extract:
    - model: Specific product/model name of the quantum computer
    - manufacturer: Company/manufacturer name
    - documentation_urls: A list of official manufacturer/product documentation URLs explicitly mentioned in the answer. Include product pages, datasheets, brochures, technical documentation pages. Do NOT invent URLs.
    - qubit_count: The number of physical qubits (text exactly as presented)
    - two_qubit_fidelity_median: The stated median two-qubit gate fidelity (text exactly as presented)
    - on_premise_deployment: Statement indicating on-premise/on-site availability
    - floor_footprint: Documented floor footprint (text exactly as presented, include units)
    - power_consumption: Documented typical power consumption (text exactly as presented, include units)
    - error_correction_support: Statement indicating native support for surface code error correction
    - commercially_available_2025: Any statement indicating commercial availability in 2025 (text exactly as presented)

    Return a JSON object with fields:
    {
      "system_a": { ... },
      "system_b": { ... }
    }

    IMPORTANT:
    - Only extract information explicitly found in the provided answer.
    - For any field not mentioned, return null (or empty list for documentation_urls).
    - For URLs, extract the actual URLs (including protocol). Accept plain URLs or markdown links, but return only the URL.
    - Preserve the text form for numbers and units as shown in the answer (e.g., "99.95%", "6.8 m^2", "25 kW", "2^20").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return urls or []


def _sys_id_str(manufacturer: Optional[str], model: Optional[str]) -> str:
    man = manufacturer.strip() if manufacturer else "unknown manufacturer"
    mod = model.strip() if model else "unknown model"
    return f"{man} {mod}"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_system_a(
    evaluator: Evaluator,
    parent_node,
    sys_a: QuantumSystemA,
) -> None:
    """
    Build the System A verification subtree (all leaves are critical).
    """
    # Parent node for System A (Critical; parallel aggregation of constraints)
    sys_a_node = evaluator.add_parallel(
        id="System_A_Ultra_High_Fidelity",
        desc="System A: ultra-high-fidelity system for error correction research meeting all stated System A constraints, with official documentation.",
        parent=parent_node,
        critical=True
    )

    # 1) Basic identification provided
    evaluator.add_custom_node(
        result=_nonempty(sys_a.model),
        id="System_A_Model_Provided",
        desc="Provides the specific quantum computer model name for System A",
        parent=sys_a_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(sys_a.manufacturer),
        id="System_A_Manufacturer_Provided",
        desc="Provides the manufacturer name for System A",
        parent=sys_a_node,
        critical=True
    )

    # 2) Documentation URL(s) are official (verifiable via URLs)
    ref_urls = _urls_or_empty(sys_a.documentation_urls)
    ref_url_node = evaluator.add_leaf(
        id="System_A_Reference_URL",
        desc="Provides an official manufacturer/product documentation URL for System A that supports verification of the required specifications",
        parent=sys_a_node,
        critical=True
    )
    claim_ref = (
        f"At least one of the provided URLs is an official manufacturer/product documentation page for {_sys_id_str(sys_a.manufacturer, sys_a.model)}."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_url_node,
        sources=ref_urls,
        additional_instruction=(
            "Treat 'official documentation' as pages hosted on the manufacturer's domain (e.g., company.com) or official "
            "product documentation/datasheet portals. Brochures, spec sheets, product pages, or technical docs are acceptable; "
            "third‑party articles/blogs are not."
        )
    )

    # 3) Commercial availability in 2025
    comm_node = evaluator.add_leaf(
        id="System_A_Commercially_Available_2025",
        desc="System A is commercially available in 2025 (as supported by official documentation)",
        parent=sys_a_node,
        critical=True
    )
    claim_comm = (
        f"The product {_sys_id_str(sys_a.manufacturer, sys_a.model)} is commercially available (offered to customers) in 2025 according to the official documentation."
    )
    await evaluator.verify(
        claim=claim_comm,
        node=comm_node,
        sources=ref_urls,
        additional_instruction=(
            "Look for signals of commercial availability (e.g., product pages offering purchase/order/access, availability statements, "
            "or announcements indicating general availability). If the documentation is clearly a product page for current customers "
            "and not just a research prototype, consider it commercially available. If the page explicitly dates availability to 2025, "
            "that strongly supports the claim."
        )
    )

    # 4) Trapped-ion qubit technology
    tech_node = evaluator.add_leaf(
        id="System_A_Qubit_Technology",
        desc="System A uses trapped-ion qubit technology",
        parent=sys_a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system uses trapped-ion qubit technology.",
        node=tech_node,
        sources=ref_urls,
        additional_instruction=(
            f"Verify the qubit technology on the documentation page(s) for {_sys_id_str(sys_a.manufacturer, sys_a.model)}. "
            "Synonyms/phrases may include 'ion trap', 'trapped ion', 'ion-based qubits'."
        )
    )

    # 5) Two-qubit gate fidelity exceeding 99.9%
    fidelity_node = evaluator.add_leaf(
        id="System_A_Two_Qubit_Fidelity",
        desc="System A achieves two-qubit gate fidelity exceeding 99.9%",
        parent=sys_a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system achieves two-qubit gate fidelity exceeding 99.9%.",
        node=fidelity_node,
        sources=ref_urls,
        additional_instruction=(
            "Accept equivalent phrasing such as '>= 99.9%', '99.9%+', or gate-specific names (CNOT/MS/CZ). "
            "If the documentation indicates at least 99.9%, this passes."
        )
    )

    # 6) At least 50 physical qubits
    qubits_node = evaluator.add_leaf(
        id="System_A_Qubit_Count",
        desc="System A has at least 50 physical qubits",
        parent=sys_a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system has at least 50 physical qubits.",
        node=qubits_node,
        sources=ref_urls,
        additional_instruction=(
            "Verify the documented number of physical qubits. If a range or '>=50' is shown, that passes."
        )
    )

    # 7) All-to-all qubit connectivity
    conn_node = evaluator.add_leaf(
        id="System_A_Connectivity",
        desc="System A provides all-to-all qubit connectivity",
        parent=sys_a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system provides all-to-all qubit connectivity.",
        node=conn_node,
        sources=ref_urls,
        additional_instruction=(
            "Look for 'all-to-all', 'full connectivity', or statements indicating any qubit can directly interact with any other."
        )
    )

    # 8) Quantum volume at least 2^20
    qv_node = evaluator.add_leaf(
        id="System_A_Quantum_Volume",
        desc="System A has demonstrated quantum volume of at least 2^20 (1,048,576)",
        parent=sys_a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system has demonstrated quantum volume of at least 2^20 (1,048,576).",
        node=qv_node,
        sources=ref_urls,
        additional_instruction=(
            "Accept exact forms '2^20' or '1,048,576'. If documentation indicates equal or greater quantum volume, it passes."
        )
    )

    # 9) Mid-circuit measurement support
    mcm_node = evaluator.add_leaf(
        id="System_A_Mid_Circuit_Measurement",
        desc="System A supports mid-circuit measurement capability",
        parent=sys_a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system supports mid-circuit measurement capability.",
        node=mcm_node,
        sources=ref_urls,
        additional_instruction=(
            "Look for 'mid-circuit measurement', 'in-circuit measurement', or equivalent capabilities (e.g., measurement and feed-forward during a circuit)."
        )
    )


async def verify_system_b(
    evaluator: Evaluator,
    parent_node,
    sys_b: QuantumSystemB,
) -> None:
    """
    Build the System B verification subtree (all leaves are critical).
    """
    # Parent node for System B (Critical; parallel aggregation of constraints)
    sys_b_node = evaluator.add_parallel(
        id="System_B_Mid_Scale_Development",
        desc="System B: mid-scale development system meeting all stated System B constraints, with official documentation.",
        parent=parent_node,
        critical=True
    )

    # 1) Basic identification provided
    evaluator.add_custom_node(
        result=_nonempty(sys_b.model),
        id="System_B_Model_Provided",
        desc="Provides the specific quantum computer model name for System B",
        parent=sys_b_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(sys_b.manufacturer),
        id="System_B_Manufacturer_Provided",
        desc="Provides the manufacturer name for System B",
        parent=sys_b_node,
        critical=True
    )

    # 2) Documentation URL(s) are official
    ref_urls = _urls_or_empty(sys_b.documentation_urls)
    ref_url_node = evaluator.add_leaf(
        id="System_B_Reference_URL",
        desc="Provides an official manufacturer/product documentation URL for System B that supports verification of the required specifications",
        parent=sys_b_node,
        critical=True
    )
    claim_ref = (
        f"At least one of the provided URLs is an official manufacturer/product documentation page for {_sys_id_str(sys_b.manufacturer, sys_b.model)}."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_url_node,
        sources=ref_urls,
        additional_instruction=(
            "Treat 'official documentation' as pages hosted on the manufacturer's domain (e.g., company.com) or official "
            "product documentation/datasheet portals. Brochures, spec sheets, product pages, or technical docs are acceptable; "
            "third‑party articles/blogs are not."
        )
    )

    # 3) Commercial availability in 2025
    comm_node = evaluator.add_leaf(
        id="System_B_Commercially_Available_2025",
        desc="System B is commercially available in 2025 (as supported by official documentation)",
        parent=sys_b_node,
        critical=True
    )
    claim_comm = (
        f"The product {_sys_id_str(sys_b.manufacturer, sys_b.model)} is commercially available (offered to customers) in 2025 according to the official documentation."
    )
    await evaluator.verify(
        claim=claim_comm,
        node=comm_node,
        sources=ref_urls,
        additional_instruction=(
            "Look for signals of commercial availability (e.g., product pages offering purchase/order/access, availability statements, "
            "or announcements indicating general availability). If the documentation is clearly a product page for current customers "
            "and not just a research prototype, consider it commercially available. If the page explicitly dates availability to 2025, "
            "that strongly supports the claim."
        )
    )

    # 4) Qubit count between 50 and 150
    qc_node = evaluator.add_leaf(
        id="System_B_Qubit_Count",
        desc="System B has between 50 and 150 physical qubits",
        parent=sys_b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system has a physical qubit count between 50 and 150 inclusive.",
        node=qc_node,
        sources=ref_urls,
        additional_instruction=(
            "Verify documented qubit count. Accept ranges or explicit statements that place the count within [50, 150]."
        )
    )

    # 5) Median two-qubit fidelity >= 99%
    fid_node = evaluator.add_leaf(
        id="System_B_Two_Qubit_Fidelity",
        desc="System B achieves median two-qubit gate fidelity of at least 99%",
        parent=sys_b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system achieves median two-qubit gate fidelity of at least 99%.",
        node=fid_node,
        sources=ref_urls,
        additional_instruction=(
            "Accept equivalent phrasing such as '>= 99%', 'about 99%', or statements clearly indicating median two-qubit fidelity at or above 99%."
        )
    )

    # 6) On-premise deployment availability
    onprem_node = evaluator.add_leaf(
        id="System_B_On_Premise_Deployment",
        desc="System B is available for on-premise deployment",
        parent=sys_b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system is available for on-premise (on-site) deployment.",
        node=onprem_node,
        sources=ref_urls,
        additional_instruction=(
            "Look for explicit mention of 'on-premise', 'on-site', or 'deployable at customer facilities'. Cloud-only offerings do not satisfy this."
        )
    )

    # 7) Floor footprint <= 10 m²
    footprint_node = evaluator.add_leaf(
        id="System_B_Floor_Footprint",
        desc="System B has a documented floor footprint not exceeding 10 square meters",
        parent=sys_b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The documented floor footprint is not exceeding 10 square meters.",
        node=footprint_node,
        sources=ref_urls,
        additional_instruction=(
            "Verify footprint values and units. Accept conversions (e.g., square feet to m²) and typical layout footprints. "
            "If the documentation states ~6.8 m² or any value ≤ 10 m², this passes."
        )
    )

    # 8) Typical power consumption <= 30 kW
    power_node = evaluator.add_leaf(
        id="System_B_Power_Consumption",
        desc="System B has documented typical power consumption not exceeding 30 kilowatts",
        parent=sys_b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The typical power consumption does not exceed 30 kilowatts.",
        node=power_node,
        sources=ref_urls,
        additional_instruction=(
            "Verify typical/nominal power consumption figures (e.g., 'typical', 'average'). Accept ≤ 30 kW."
        )
    )

    # 9) Native support for surface code error correction
    ec_node = evaluator.add_leaf(
        id="System_B_Error_Correction_Support",
        desc="System B natively supports surface code error correction architecture",
        parent=sys_b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system natively supports the surface code error correction architecture.",
        node=ec_node,
        sources=ref_urls,
        additional_instruction=(
            "Look for explicit mention of 'surface code', 'native surface code support', or equivalent technical statements (e.g., stabilizer measurement, lattice surgery) "
            "indicating native support, not just research compatibility."
        )
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the dual-system quantum infrastructure task and return a structured result dictionary.
    """
    # Initialize evaluator (root node is created non-critical by design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Two systems verified independently
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

    # Create a top-level critical node to represent the facility requirement aggregation
    infra_node = evaluator.add_parallel(
        id="Research_Facility_Quantum_Infrastructure",
        desc="Identify two commercial quantum computer systems available in 2025 (System A and System B) and provide model, manufacturer, and official documentation URLs confirming required specifications.",
        parent=root,
        critical=True
    )

    # Extract the system details from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_dual_systems(),
        template_class=DualSystemExtraction,
        extraction_name="dual_systems_extraction"
    )

    # Ensure we have objects to work with even if extraction returns None
    sys_a = extraction.system_a or QuantumSystemA()
    sys_b = extraction.system_b or QuantumSystemB()

    # Optional: record custom info for transparency
    evaluator.add_custom_info(
        info={
            "system_a_summary": {
                "model": sys_a.model,
                "manufacturer": sys_a.manufacturer,
                "doc_urls_count": len(sys_a.documentation_urls),
            },
            "system_b_summary": {
                "model": sys_b.model,
                "manufacturer": sys_b.manufacturer,
                "doc_urls_count": len(sys_b.documentation_urls),
            }
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    # Build verification subtrees for System A and System B
    await verify_system_a(evaluator, infra_node, sys_a)
    await verify_system_b(evaluator, infra_node, sys_b)

    # Return the aggregated summary
    return evaluator.get_summary()